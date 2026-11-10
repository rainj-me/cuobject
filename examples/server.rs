// Copyright 2026 Rain Jiang
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! A minimal S3-over-RDMA object server, keeping objects in memory.
//!
//! Requests arrive over plain HTTP (see `common/http.rs`); object bytes move
//! over RDMA between a per-connection staging buffer and the client memory
//! the request's token describes. Run it on a host whose RDMA NIC owns
//! `--rdma-addr`, then point `examples/client.rs` at `--listen`:
//!
//! ```text
//! cargo run --release --example server -- --rdma-addr 10.0.0.1 --listen 0.0.0.0:8080
//! ```

#[path = "common/http.rs"]
mod http;

use cuobject::{
    BYTES_TRANSFERRED_HEADER, Channel, REPLY_HEADER, RdmaToken, RegisteredBuffer, Server,
    ServerOptions, Source, TOKEN_HEADER,
};
use http::Head;
use std::collections::HashMap;
use std::error::Error;
use std::io::{BufReader, BufWriter};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex, PoisonError};

type Objects = Mutex<HashMap<String, Vec<u8>>>;

struct Args {
    rdma_addr: IpAddr,
    rdma_port: u16,
    listen: SocketAddr,
    staging: usize,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut args = Args {
        rdma_addr: IpAddr::from([0, 0, 0, 0]),
        rdma_port: cuobject::DEFAULT_PORT,
        listen: "0.0.0.0:8080".parse()?,
        staging: 64 << 20,
    };
    let mut rdma_addr = None;
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let value = argv.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--rdma-addr" => rdma_addr = Some(value.parse()?),
            "--rdma-port" => args.rdma_port = value.parse()?,
            "--listen" => args.listen = value.parse()?,
            "--staging-mib" => args.staging = value.parse::<usize>()? << 20,
            _ => return Err(format!("unknown flag {flag}").into()),
        }
    }
    args.rdma_addr = rdma_addr.ok_or(
        "usage: server --rdma-addr <ip of an RDMA NIC> [--rdma-port 18515] \
         [--listen 0.0.0.0:8080] [--staging-mib 64]",
    )?;
    Ok(args)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let (major, minor) = cuobject::header_version();
    // Shared by every connection thread; channels and staging buffers keep it
    // alive through their own `Arc`.
    let server = Arc::new(Server::new(
        args.rdma_addr,
        args.rdma_port,
        &ServerOptions::default(),
    )?);
    let listener = TcpListener::bind(args.listen)?;
    println!(
        "cuObject server {major}.{minor} on {}:{}, HTTP on {}, {} MiB staging per connection",
        args.rdma_addr,
        args.rdma_port,
        listener.local_addr()?,
        args.staging >> 20
    );

    let objects: Arc<Objects> = Arc::default();
    for stream in listener.incoming() {
        let stream = stream?;
        let peer = stream.peer_addr()?;
        let server = Arc::clone(&server);
        let objects = Arc::clone(&objects);
        std::thread::spawn(move || {
            println!("{peer}: connected");
            match serve(server, &objects, stream, args.staging) {
                Ok(()) => println!("{peer}: closed"),
                Err(err) => eprintln!("{peer}: {err}"),
            }
        });
    }
    Ok(())
}

/// Serves one connection until the client closes it.
fn serve(
    server: Arc<Server>,
    objects: &Objects,
    stream: TcpStream,
    staging_len: usize,
) -> Result<(), Box<dyn Error>> {
    // The connection's own channel and staging buffer: `'static` handles,
    // since they hold the server through an `Arc` rather than a borrow.
    let mut channel: Channel<Arc<Server>> = Channel::lease(Arc::clone(&server))?;
    let staging: RegisteredBuffer<Arc<Server>> = RegisteredBuffer::new(server, staging_len)?;
    let mut input = BufReader::new(stream.try_clone()?);
    let mut output = BufWriter::new(stream);
    while let Some(request) = http::read_head(&mut input)? {
        let response = match handle(&request, &mut channel, &staging, objects) {
            Ok(moved) => {
                println!("{} {} -> {moved} bytes", request.line, range_of(&request));
                reply(200, "OK", moved, None)
            }
            Err(failure) => {
                eprintln!("{} -> {} {}", request.line, failure.status, failure.detail);
                reply(failure.status, failure.reason, 0, Some(failure.detail))
            }
        };
        response.write_to(&mut output)?;
    }
    Ok(())
}

/// Moves one chunk and returns the bytes moved.
fn handle(
    request: &Head,
    channel: &mut Channel<Arc<Server>>,
    staging: &RegisteredBuffer<Arc<Server>>,
    objects: &Objects,
) -> Result<usize, Failure> {
    let mut words = request.line.split(' ');
    let method = words.next().unwrap_or_default();
    let key = words
        .next()
        .and_then(|path| path.strip_prefix('/'))
        .filter(|key| !key.is_empty())
        .ok_or_else(|| Failure::bad_request("missing object key"))?;
    let token = request
        .header(TOKEN_HEADER)
        .ok_or_else(|| Failure::bad_request("missing x-amz-rdma-token"))?;
    let token = RdmaToken::parse(token).map_err(|err| Failure::bad_request(err.to_string()))?;
    let (offset, len) = request
        .header("range")
        .and_then(http::parse_range)
        .ok_or_else(|| Failure::bad_request("missing or malformed range"))?;
    let (offset, len) = (offset as usize, len as usize);
    if len > staging.len() {
        return Err(Failure {
            status: 413,
            reason: "Content Too Large",
            detail: format!(
                "chunk of {len} bytes exceeds the {} byte staging buffer",
                staging.len()
            ),
        });
    }

    match method {
        "GET" => {
            {
                let objects = objects.lock().unwrap_or_else(PoisonError::into_inner);
                let object = objects.get(key).ok_or(Failure {
                    status: 404,
                    reason: "Not Found",
                    detail: format!("no object {key}"),
                })?;
                let end = offset
                    .checked_add(len)
                    .filter(|&end| end <= object.len())
                    .ok_or(Failure {
                        status: 416,
                        reason: "Range Not Satisfiable",
                        detail: format!("object {key} has {} bytes", object.len()),
                    })?;
                // SAFETY: this thread owns `staging` and no transfer runs on it.
                unsafe { staging.slice_mut(0, len) }?.copy_from_slice(&object[offset..end]);
            }
            let source = Source::Range {
                buffer: staging,
                offset: 0,
                len,
            };
            Ok(channel.get(key, &token, 0, source)?)
        }
        "PUT" => {
            let moved = channel.put(key, &token, 0, staging, 0, len)?;
            let mut objects = objects.lock().unwrap_or_else(PoisonError::into_inner);
            let object = objects.entry(key.to_owned()).or_default();
            if object.len() < offset + moved {
                object.resize(offset + moved, 0);
            }
            // SAFETY: the PUT finished and this thread owns `staging`.
            object[offset..offset + moved].copy_from_slice(unsafe { staging.slice(0, moved) }?);
            Ok(moved)
        }
        _ => Err(Failure {
            status: 405,
            reason: "Method Not Allowed",
            detail: format!("{method} is not GET or PUT"),
        }),
    }
}

fn range_of(request: &Head) -> &str {
    request.header("range").unwrap_or("-")
}

fn reply(status: u16, reason: &str, moved: usize, detail: Option<String>) -> Head {
    let mut headers = vec![
        (REPLY_HEADER.to_owned(), status.to_string()),
        (BYTES_TRANSFERRED_HEADER.to_owned(), moved.to_string()),
    ];
    if let Some(detail) = detail {
        headers.push(("x-cuobject-error".to_owned(), detail));
    }
    Head {
        line: format!("HTTP/1.1 {status} {reason}"),
        headers,
    }
}

struct Failure {
    status: u16,
    reason: &'static str,
    detail: String,
}

impl Failure {
    fn bad_request(detail: impl Into<String>) -> Self {
        Self {
            status: 400,
            reason: "Bad Request",
            detail: detail.into(),
        }
    }
}

impl From<cuobject::Error> for Failure {
    fn from(err: cuobject::Error) -> Self {
        Self {
            status: 500,
            reason: "Internal Server Error",
            detail: err.to_string(),
        }
    }
}
