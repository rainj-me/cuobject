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

//! Runs against the real library and an RDMA device; no client is needed.
//!
//! `CUOBJECT_TEST_ADDR=<address of an RDMA NIC> cargo test -- --ignored`

#![cfg(feature = "server")]

use cuobject::{
    Channel, Completion, Error, RdmaToken, RegisteredBuffer, Server, ServerOptions, SgEntry, Source,
};
use std::sync::Arc;

const TOKEN: &str =
    "0000f52eddc00000:00001000:001824d2:0000:002873:1:00000000000000000000ffff0a0d5410";

fn start_server(port: u16) -> Server {
    let addr = std::env::var("CUOBJECT_TEST_ADDR")
        .expect("set CUOBJECT_TEST_ADDR to the address of an RDMA NIC");
    Server::new(
        addr.parse().expect("address"),
        port,
        &ServerOptions::default(),
    )
    .expect("server")
}

#[test]
#[ignore = "needs an RDMA device and CUOBJECT_TEST_ADDR"]
fn buffers_lists_and_channels_against_the_library() {
    assert!(cuobject::header_version() >= (2, 0));
    let server = start_server(18615);

    assert!(matches!(
        server.register_host_buffer(0),
        Err(Error::InvalidArgument(_))
    ));
    let buffer = server.register_host_buffer(64 * 1024).expect("register");
    assert_eq!(buffer.as_mut_ptr() as usize % 4096, 0);
    // SAFETY: no transfer runs and no other slice is live.
    unsafe {
        assert!(
            buffer
                .slice(0, buffer.len())
                .expect("whole")
                .iter()
                .all(|&b| b == 0)
        );
        buffer
            .slice_mut(4096, 5)
            .expect("range")
            .copy_from_slice(b"hello");
        assert_eq!(buffer.slice(4096, 5).expect("range"), b"hello");
        assert!(buffer.slice(buffer.len(), 1).is_err());
    }

    assert!(buffer.sg_buffer(&[]).is_err());
    assert!(
        buffer
            .sg_buffer(&[SgEntry {
                offset: 60 * 1024,
                size: 8 * 1024
            }])
            .is_err()
    );
    let list = buffer
        .sg_buffer(&[
            SgEntry {
                offset: 0,
                size: 4096,
            },
            SgEntry {
                offset: 8192,
                size: 100,
            },
        ])
        .expect("list");
    assert_eq!(list.len(), 4196);

    let mut first = server.channel().expect("channel");
    let second = server.channel().expect("channel");
    assert_ne!(first.id(), second.id());

    // Rejected before any DMA.
    let token = RdmaToken::parse(TOKEN).expect("token");
    let too_long = Source::Range {
        buffer: &buffer,
        offset: 0,
        len: 4097,
    };
    assert!(matches!(
        first.get("k", &token, 0, too_long),
        Err(Error::TokenTooSmall { needed: 4097, .. })
    ));
    assert!(matches!(
        first.get("k", &token, 0, Source::List(&list)),
        Err(Error::TokenTooSmall { needed: 4196, .. })
    ));
    let empty = Source::Range {
        buffer: &buffer,
        offset: 0,
        len: 0,
    };
    assert!(matches!(
        first.get("k", &token, 0, empty),
        Err(Error::InvalidArgument(_))
    ));
    let other_server = start_server(18616);
    let other = other_server.register_host_buffer(4096).expect("register");
    let foreign = Source::Range {
        buffer: &other,
        offset: 0,
        len: 16,
    };
    assert!(matches!(
        first.get("k", &token, 0, foreign),
        Err(Error::InvalidArgument(_))
    ));

    let fd = first.completion_fd().expect("completion descriptor");
    assert!(std::os::fd::AsRawFd::as_raw_fd(&fd) >= 0);
    let mut events = [Completion::default(); 4];
    assert_eq!(first.process_completions(&mut events).expect("drain"), 0);
}

#[test]
#[ignore = "needs an RDMA device and CUOBJECT_TEST_ADDR"]
fn arc_handles_keep_the_server_alive() {
    let server = Arc::new(start_server(18617));
    let buffer = RegisteredBuffer::new(Arc::clone(&server), 4096).expect("register");
    let list = buffer
        .sg_buffer(&[SgEntry {
            offset: 0,
            size: 4096,
        }])
        .expect("list");
    let mut channel = Channel::lease(Arc::clone(&server)).expect("channel");
    let fd = channel.completion_fd().expect("completion descriptor");

    // A borrowed buffer works with an owning channel on the same server.
    let borrowed = server.register_host_buffer(4096).expect("register");
    let token = RdmaToken::parse(TOKEN).expect("token");
    let too_long = Source::Range {
        buffer: &borrowed,
        offset: 0,
        len: 4096,
    };
    assert!(matches!(
        channel.get("k", &token, 1, too_long),
        Err(Error::TokenTooSmall { needed: 4097, .. })
    ));
    drop(borrowed);

    // Owning handles keep the library object alive whatever the drop order.
    drop(server);
    drop(fd);
    drop(channel);
    drop(list);
    drop(buffer);
}
