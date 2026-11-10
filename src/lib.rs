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

//! Safe bindings to NVIDIA's cuObject libraries for S3-over-RDMA (GPUDirect
//! object loading).
//!
//! A client registers a buffer in its (host or GPU) memory and sends an
//! ordinary S3 request with a [`TOKEN_HEADER`] header describing it. Instead
//! of carrying the body over HTTP, the server moves the bytes with RDMA: a GET
//! writes them from server memory into the client buffer, a PUT reads them
//! from the client buffer. The HTTP response then carries [`REPLY_HEADER`] and
//! [`BYTES_TRANSFERRED_HEADER`]. This crate covers both ends; the HTTP
//! exchange in between belongs to the application (`examples/server.rs` and
//! `examples/client.rs` show a minimal one).
//!
//! | Feature | Library | Types |
//! |---|---|---|
//! | `server` (default) | `libcuobjserver` | [`Server`], [`RegisteredBuffer`], [`Channel`] |
//! | `client` | `libcuobjclient`, `libcufile` | [`Client`], [`ClientBuffer`] |
//!
//! # Server
//!
//! - [`Server`] binds the RDMA device that owns an address.
//! - [`RegisteredBuffer::new`] maps zero-filled, page-aligned host memory and
//!   registers it. [`RegisteredBuffer::sg_buffer`] builds a scatter-gather
//!   list over several ranges of one buffer.
//! - [`Channel::lease`] leases a channel. The library forbids using one
//!   channel from two threads at once, so transfers take `&mut Channel`.
//! - [`Channel::get`] and [`Channel::put`] block until the RDMA operation
//!   completed. [`Channel::get_async`] submits and returns; completions arrive
//!   on [`Channel::completion_fd`] and are drained with
//!   [`Channel::process_completions`].
//!
//! # Client
//!
//! - [`Client`] connects to the RDMA devices the cuFile configuration names.
//! - [`ClientBuffer::new`] maps zero-filled, page-aligned host memory and
//!   registers it; [`ClientBuffer::from_raw`] registers memory the caller
//!   owns, such as a CUDA allocation.
//! - [`ClientBuffer::get`] and [`ClientBuffer::put`] run a transfer: the
//!   library calls a closure for each chunk with its token ([`Chunk::token`]);
//!   the closure sends the request to the server and returns the bytes the
//!   server moved. [`ClientBuffer::token`] makes a token for a transfer driven
//!   without the callback.
//!
//! # Ownership
//!
//! Server-side handles hold their server through a [`ServerRef`], client-side
//! buffers their client through a [`ClientRef`], chosen by the caller: `&T`
//! borrows it, `Arc<T>` or `Rc<T>` share ownership of it. Either way the
//! server or client outlives every handle, and the crate itself never
//! allocates a reference count. Scatter-gather lists and client tokens borrow
//! their buffer.
//!
//! # Memory access
//!
//! The NIC reads and writes registered memory outside Rust's view. Reading or
//! writing it from Rust is therefore `unsafe` ([`RegisteredBuffer::slice`],
//! [`ClientBuffer::slice_mut`] and so on): the caller makes sure no transfer
//! and no other reference touches the same range at the same time.

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "server")]
mod buffer;
#[cfg(feature = "client")]
mod client;
#[cfg(feature = "server")]
mod server;
mod token;

#[cfg(feature = "server")]
pub use buffer::{RegisteredBuffer, SgBuffer, SgEntry};
#[cfg(feature = "client")]
pub use client::{
    Chunk, Client, ClientBuffer, ClientRef, ClientToken, MemoryType, Operation, client_version,
};
#[cfg(feature = "server")]
pub use server::{
    Channel, Completion, CompletionFd, Server, ServerOptions, ServerRef, Source, header_version,
};
pub use token::{
    BYTES_TRANSFERRED_HEADER, DEFAULT_MAX_SG_ENTRIES, DEFAULT_PORT, MAX_TRANSFER_BYTES,
    REPLY_HEADER, REPLY_NOT_IMPLEMENTED, RdmaToken, TOKEN_HEADER,
};

use std::io;

/// Errors from the library or from the argument checks made before calling it.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A library call failed.
    #[error("cuObject {op} failed: {source}")]
    Library {
        /// The operation that failed.
        op: &'static str,
        /// The errno the library returned.
        #[source]
        source: io::Error,
    },
    /// An RDMA transfer failed.
    #[error("cuObject transfer failed (verbs status {verbs_status}): {source}")]
    Transfer {
        /// The `ibv_wc_status` of the work completion (`0` when unknown).
        verbs_status: i32,
        /// The errno the library returned (`EPROTO` for a malformed or
        /// mismatched token).
        #[source]
        source: io::Error,
    },
    /// An argument was rejected before the library saw it.
    #[error("invalid argument: {0}")]
    InvalidArgument(&'static str),
    /// An `x-amz-rdma-token` value could not be parsed.
    #[error("malformed RDMA token: {0}")]
    InvalidToken(&'static str),
    /// The transfer would run past the end of the client's buffer.
    #[error("the client buffer ({buffer} bytes) ends before the transfer ({needed} bytes)")]
    TokenTooSmall {
        /// Length of the client buffer from the token.
        buffer: u64,
        /// Offset of the transfer's end in the client buffer.
        needed: u64,
    },
    /// The library has no completion descriptor for the channel.
    #[error("no completion descriptor for channel {0}")]
    AsyncUnavailable(u16),
}

/// `Result` with [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

const EIO: i32 = 5;
#[cfg(feature = "client")]
const EINVAL: i32 = 22;
#[cfg(feature = "client")]
const EPROTO: i32 = 71;

/// The error for a negative errno return.
#[allow(dead_code, reason = "unused with no feature enabled")]
fn errno(rc: isize) -> io::Error {
    io::Error::from_raw_os_error(i32::try_from(rc.unsigned_abs()).unwrap_or(EIO))
}

#[allow(dead_code, reason = "unused with no feature enabled")]
fn library_error(op: &'static str, rc: isize) -> Error {
    Error::Library {
        op,
        source: errno(rc),
    }
}
