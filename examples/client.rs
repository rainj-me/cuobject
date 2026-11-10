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

//! PUTs an object to `examples/server.rs` over RDMA, GETs it back into the
//! same (cleared) buffer and checks every byte.
//!
//! The cuObject client finds its RDMA NICs through the cuFile configuration;
//! point `CUFILE_ENV_PATH_JSON` at a `cufile.json` that lists them (see
//! `examples/cufile.json`). Without `nvidia_peermem` (open kernel modules on
//! the kernel's own RDMA stack), cuFile needs `CUFILE_DMABUF_ENABLE=true`:
//!
//! ```text
//! CUFILE_DMABUF_ENABLE=true CUFILE_ENV_PATH_JSON=examples/cufile.json \
//!   cargo run --release --features client --example client -- --server 10.0.0.1:8080
//! ```

#[path = "common/http.rs"]
mod http;

use cuobject::{BYTES_TRANSFERRED_HEADER, Chunk, Client, REPLY_HEADER, TOKEN_HEADER};
use http::Head;
use std::error::Error;
use std::io::{self, BufReader, BufWriter};
use std::net::TcpStream;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

struct Args {
    server: String,
    key: String,
    size: usize,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut server = None;
    let mut args = Args {
        server: String::new(),
        key: "demo/object".to_owned(),
        size: 16 << 20,
    };
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let value = argv.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--server" => server = Some(value),
            "--key" => args.key = value,
            "--size-mib" => args.size = value.parse::<usize>()? << 20,
            _ => return Err(format!("unknown flag {flag}").into()),
        }
    }
    args.server =
        server.ok_or("usage: client --server <host:port> [--key demo/object] [--size-mib 16]")?;
    if args.size == 0 {
        return Err("--size-mib must be positive".into());
    }
    Ok(args)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let (major, minor) = cuobject::client_version();
    let client = Client::new()?;
    let buffer = client.register_host_buffer(args.size)?;
    println!(
        "cuObject client {major}.{minor}: registered {} MiB of {:?} memory, chunks up to {} bytes",
        args.size >> 20,
        Client::memory_type(buffer.as_mut_ptr()),
        buffer.max_request_size()?
    );

    // The library may call back for several chunks at once, from its own
    // threads: each takes its own connection from the pool, so chunks run in
    // parallel (and the server serves each connection on its own channel).
    let pool = Pool::new(&args.server)?;
    let send = |method: &str, chunk: Chunk<'_>| pool.send(method, &args.key, chunk);

    // SAFETY: no transfer runs on `buffer` and nothing else borrows it.
    for (i, byte) in unsafe { buffer.slice_mut(0, args.size) }?
        .iter_mut()
        .enumerate()
    {
        *byte = pattern(i);
    }
    let start = Instant::now();
    let put = buffer.put(0, args.size, |chunk| send("PUT", chunk))?;
    report("PUT", put, start, &pool);

    // SAFETY: the PUT finished; nothing else borrows `buffer`.
    unsafe { buffer.slice_mut(0, args.size) }?.fill(0);
    let start = Instant::now();
    let got = buffer.get(0, args.size, |chunk| send("GET", chunk))?;
    report("GET", got, start, &pool);

    // SAFETY: the GET finished; nothing else borrows `buffer`.
    let data = unsafe { buffer.slice(0, args.size) }?;
    if let Some(i) = (0..args.size).find(|&i| data[i] != pattern(i)) {
        return Err(format!("byte {i} is {:#04x}, expected {:#04x}", data[i], pattern(i)).into());
    }
    println!(
        "verified {} bytes of {} round-tripped over RDMA",
        args.size, args.key
    );
    Ok(())
}

/// The object's content: varies within and across pages.
fn pattern(i: usize) -> u8 {
    (i % 251) as u8 ^ (i >> 12) as u8
}

fn report(op: &str, bytes: usize, start: Instant, pool: &Pool) {
    let secs = start.elapsed().as_secs_f64();
    println!(
        "{op}: {bytes} bytes in {:.3} ms ({:.2} GiB/s), {} chunks, {} connections open",
        secs * 1e3,
        bytes as f64 / secs / f64::from(1u32 << 30),
        pool.chunks.swap(0, Ordering::Relaxed),
        pool.opened.load(Ordering::Relaxed)
    );
}

/// Idle keep-alive connections to the server. The lock only guards taking or
/// returning one, never a request, so concurrent chunks don't wait on each
/// other's I/O; a chunk that finds no idle connection opens another.
struct Pool {
    host: String,
    idle: Mutex<Vec<Connection>>,
    opened: AtomicUsize,
    chunks: AtomicUsize,
}

impl Pool {
    fn new(host: &str) -> io::Result<Self> {
        // Fail fast if the server is unreachable.
        let first = Connection::open(host)?;
        Ok(Self {
            host: host.to_owned(),
            idle: Mutex::new(vec![first]),
            opened: AtomicUsize::new(1),
            chunks: AtomicUsize::new(0),
        })
    }

    fn send(&self, method: &str, key: &str, chunk: Chunk<'_>) -> io::Result<usize> {
        self.chunks.fetch_add(1, Ordering::Relaxed);
        let idle = self
            .idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop();
        let mut connection = match idle {
            Some(connection) => connection,
            None => {
                self.opened.fetch_add(1, Ordering::Relaxed);
                Connection::open(&self.host)?
            }
        };
        // On error the connection may be mid-response; drop it rather than
        // return it to the pool.
        let moved = connection.send(method, key, chunk)?;
        self.idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(connection);
        Ok(moved)
    }
}

/// One keep-alive HTTP connection to the server.
struct Connection {
    host: String,
    input: BufReader<TcpStream>,
    output: BufWriter<TcpStream>,
}

impl Connection {
    fn open(host: &str) -> io::Result<Self> {
        let stream = TcpStream::connect(host)?;
        stream.set_nodelay(true)?;
        Ok(Self {
            host: host.to_owned(),
            input: BufReader::new(stream.try_clone()?),
            output: BufWriter::new(stream),
        })
    }

    /// Asks the server to move `chunk` and returns the bytes it moved.
    fn send(&mut self, method: &str, key: &str, chunk: Chunk<'_>) -> io::Result<usize> {
        Head {
            line: format!("{method} /{key} HTTP/1.1"),
            headers: vec![
                ("host".to_owned(), self.host.clone()),
                (TOKEN_HEADER.to_owned(), chunk.token().to_owned()),
                (
                    "range".to_owned(),
                    http::range(chunk.offset(), chunk.len() as u64),
                ),
            ],
        }
        .write_to(&mut self.output)?;

        let response = http::read_head(&mut self.input)?
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "server closed"))?;
        if response.line.split(' ').nth(1) != Some("200")
            || response.header(REPLY_HEADER) != Some("200")
        {
            return Err(io::Error::other(format!(
                "{method} {}: {} ({})",
                http::range(chunk.offset(), chunk.len() as u64),
                response.line,
                response.header("x-cuobject-error").unwrap_or("no detail")
            )));
        }
        response
            .header(BYTES_TRANSFERRED_HEADER)
            .and_then(|moved| moved.parse().ok())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing byte count"))
    }
}
