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

//! The server, its channels and transfers.

use crate::buffer::{RegisteredBuffer, SgBuffer};
use crate::{Error, RdmaToken, Result, errno, library_error};
use cuobject_sys as sys;
use std::ffi::{CString, c_int};
use std::fmt;
use std::io;
use std::mem::{align_of, offset_of, size_of};
use std::net::IpAddr;
use std::num::NonZeroUsize;
use std::ops::Deref;
use std::os::fd::{AsRawFd, RawFd};
use std::ptr::{self, NonNull};
use std::rc::Rc;
use std::sync::Arc;

/// Library tuning; `0` keeps the library default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ServerOptions {
    /// DC initiators, which bounds the channels that can be leased at once.
    pub num_dcis: u32,
    /// Completion queue depth.
    pub cq_depth: u32,
    /// Most scatter-gather entries per list.
    pub max_sge: u32,
}

/// Version of `cuobjserver.h` this crate was compiled against, as
/// `(major, minor)`.
pub fn header_version() -> (u32, u32) {
    // SAFETY: a pure function of compile-time constants.
    let packed = unsafe { sys::cuobj_version() };
    let packed = u32::try_from(packed).unwrap_or(0);
    (packed / 1000, packed % 1000)
}

/// A cuObject server bound to the RDMA device that owns its address,
/// destroyed on drop.
///
/// Every handle created from it holds it through a [`ServerRef`], so the
/// server outlives them all.
///
/// # Examples
///
/// Serving a GET from a borrowed server:
///
/// ```no_run
/// use cuobject::{RdmaToken, Server, ServerOptions, Source};
///
/// # fn main() -> cuobject::Result<()> {
/// let server = Server::new("10.0.0.1".parse().unwrap(), cuobject::DEFAULT_PORT, &ServerOptions::default())?;
/// let staging = server.register_host_buffer(1 << 20)?;
/// // SAFETY: no transfer uses `staging` yet and nothing else borrows it.
/// unsafe { staging.slice_mut(0, 5)? }.copy_from_slice(b"hello");
///
/// // The value of the request's `x-amz-rdma-token` header.
/// let token = RdmaToken::parse("0000f52eddc00000:00100000:001824d2:0000:002873:1:00000000000000000000ffff0a0d5410")?;
/// let mut channel = server.channel()?;
/// let moved = channel.get("bucket/key", &token, 0, Source::Range { buffer: &staging, offset: 0, len: 5 })?;
/// assert_eq!(moved, 5);
/// # Ok(())
/// # }
/// ```
///
/// Owned, `'static` handles that can live next to the server in a pool or
/// move into spawned tasks:
///
/// ```no_run
/// use std::sync::Arc;
/// use cuobject::{Channel, RegisteredBuffer, Server, ServerOptions};
///
/// # fn main() -> cuobject::Result<()> {
/// let server = Arc::new(Server::new("10.0.0.1".parse().unwrap(), cuobject::DEFAULT_PORT, &ServerOptions::default())?);
/// let staging: RegisteredBuffer<Arc<Server>> = RegisteredBuffer::new(Arc::clone(&server), 1 << 20)?;
/// let channel: Channel<Arc<Server>> = Channel::lease(Arc::clone(&server))?;
/// # Ok(())
/// # }
/// ```
pub struct Server {
    raw: NonNull<sys::cuobj_server>,
    addr: IpAddr,
    port: u16,
}

// SAFETY: the library object is thread-safe except that one channel must not
// be used by two threads at once, which `Channel` enforces with `&mut self`.
unsafe impl Send for Server {}
unsafe impl Sync for Server {}

impl Drop for Server {
    fn drop(&mut self) {
        // SAFETY: every buffer, list, channel and completion descriptor holds
        // this server through a `ServerRef`, so none is left.
        unsafe { sys::cuobj_server_destroy(self.raw.as_ptr()) };
    }
}

impl Server {
    /// Creates a server on the RDMA device that owns `addr`, listening for
    /// connection setup on `port` ([`DEFAULT_PORT`](crate::DEFAULT_PORT) is
    /// the library's default).
    pub fn new(addr: IpAddr, port: u16, options: &ServerOptions) -> Result<Self> {
        let ip = CString::new(addr.to_string())
            .map_err(|_| Error::InvalidArgument("address contains NUL"))?;
        let num_dcis = c_int::try_from(options.num_dcis)
            .map_err(|_| Error::InvalidArgument("num_dcis is too large"))?;
        let max_sge = c_int::try_from(options.max_sge)
            .map_err(|_| Error::InvalidArgument("max_sge is too large"))?;
        let mut out = ptr::null_mut();
        // SAFETY: `ip` is NUL-terminated and `out` is a valid out-pointer.
        let rc = unsafe {
            sys::cuobj_server_create(
                ip.as_ptr(),
                port,
                num_dcis,
                options.cq_depth,
                max_sge,
                &mut out,
            )
        };
        if rc < 0 {
            return Err(library_error("create server", rc as isize));
        }
        let raw =
            NonNull::new(out).ok_or(Error::InvalidArgument("library returned no server handle"))?;
        Ok(Self { raw, addr, port })
    }

    /// The address the server was created on.
    pub fn addr(&self) -> IpAddr {
        self.addr
    }

    /// The connection setup port.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// [`RegisteredBuffer::new`] on a borrow of this server.
    pub fn register_host_buffer(&self, len: usize) -> Result<RegisteredBuffer<&Self>> {
        RegisteredBuffer::new(self, len)
    }

    /// [`Channel::lease`] on a borrow of this server.
    pub fn channel(&self) -> Result<Channel<&Self>> {
        Channel::lease(self)
    }

    pub(crate) fn as_ptr(&self) -> *mut sys::cuobj_server {
        self.raw.as_ptr()
    }
}

impl fmt::Debug for Server {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Server")
            .field("addr", &self.addr)
            .field("port", &self.port)
            .finish()
    }
}

/// How a handle holds the [`Server`] it came from: `&Server` borrows it,
/// `Arc<Server>` or `Rc<Server>` share ownership of it. The handle keeps the
/// server alive through it, so the caller chooses the ownership and the crate
/// never allocates a reference count.
///
/// Sealed: the handles rely on every `deref` returning the same server.
pub trait ServerRef: Deref<Target = Server> + Clone + sealed::Sealed {}

impl ServerRef for &Server {}
impl ServerRef for Arc<Server> {}
impl ServerRef for Rc<Server> {}

mod sealed {
    use super::Server;
    use std::rc::Rc;
    use std::sync::Arc;

    pub trait Sealed {}

    impl Sealed for &Server {}
    impl Sealed for Arc<Server> {}
    impl Sealed for Rc<Server> {}
}

/// The local side of a transfer.
#[derive(Debug)]
pub enum Source<'a, S: ServerRef> {
    /// `len` bytes at `offset` of a registered buffer.
    Range {
        /// The buffer.
        buffer: &'a RegisteredBuffer<S>,
        /// Byte offset in the buffer.
        offset: usize,
        /// Length in bytes.
        len: usize,
    },
    /// Every range of a scatter-gather list, in order.
    List(&'a SgBuffer<'a, S>),
}

// Not derived: that would require `S: Copy`, which `Arc` is not.
impl<S: ServerRef> Clone for Source<'_, S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: ServerRef> Copy for Source<'_, S> {}

/// A leased channel, returned to the server on drop.
///
/// The library forbids using a channel from two threads at once; every
/// operation takes `&mut self`.
pub struct Channel<S: ServerRef> {
    server: S,
    id: u16,
}

impl<S: ServerRef> Channel<S> {
    /// Leases a channel from `server`. Fails with `EBUSY` when every channel
    /// is leased. [`Server::channel`] is the shorthand for a borrowed server.
    pub fn lease(server: S) -> Result<Self> {
        let mut id = 0u16;
        // SAFETY: the server is live and `id` is a valid out-pointer.
        let rc = unsafe { sys::cuobj_channel_alloc(server.as_ptr(), &mut id) };
        if rc < 0 {
            return Err(library_error("lease channel", rc as isize));
        }
        Ok(Self { server, id })
    }

    /// The library's channel number.
    pub fn id(&self) -> u16 {
        self.id
    }

    /// RDMA-writes `source` into the client buffer described by `token`,
    /// starting `remote_offset` bytes into it, and waits for the write.
    /// Returns the bytes moved.
    pub fn get<B: ServerRef>(
        &mut self,
        key: &str,
        token: &RdmaToken,
        remote_offset: u64,
        source: Source<'_, B>,
    ) -> Result<usize> {
        let call = self.prepare(key, token, remote_offset, source)?;
        let mut verbs_status: c_int = 0;
        // SAFETY: the server, registration and handle are live for the call
        // (held by `self` and `source`); the local range lies inside the
        // registration (`prepare`); the call is synchronous.
        let rc = unsafe {
            sys::cuobj_get(
                self.server.as_ptr(),
                call.key.as_ptr(),
                call.buffer.as_ptr(),
                call.local_offset,
                call.remote_addr,
                call.len,
                call.descriptor.as_ptr(),
                self.id,
                &mut verbs_status,
            )
        };
        transfer_result(rc, verbs_status)
    }

    /// RDMA-reads `len` bytes from the client buffer described by `token`,
    /// starting `remote_offset` bytes into it, into `buffer` at `offset`, and
    /// waits for the read. Returns the bytes moved.
    pub fn put<B: ServerRef>(
        &mut self,
        key: &str,
        token: &RdmaToken,
        remote_offset: u64,
        buffer: &RegisteredBuffer<B>,
        offset: usize,
        len: usize,
    ) -> Result<usize> {
        let call = self.prepare(
            key,
            token,
            remote_offset,
            Source::Range {
                buffer,
                offset,
                len,
            },
        )?;
        let mut verbs_status: c_int = 0;
        // SAFETY: as in `get`; the library writes only the checked range.
        let rc = unsafe {
            sys::cuobj_put(
                self.server.as_ptr(),
                call.key.as_ptr(),
                call.buffer.as_ptr(),
                call.local_offset,
                call.remote_addr,
                call.len,
                call.descriptor.as_ptr(),
                self.id,
                &mut verbs_status,
            )
        };
        transfer_result(rc, verbs_status)
    }

    /// Submits the GET of [`get`](Self::get) and returns without waiting.
    /// Its [`Completion`] carries `handle` and arrives through
    /// [`process_completions`](Self::process_completions) on this channel.
    ///
    /// # Safety
    ///
    /// Until a completion carrying `handle` was drained from this channel:
    /// the buffer behind `source` (and the list, for [`Source::List`]) must
    /// stay alive, no [`RegisteredBuffer::slice_mut`] may cover the source
    /// range, and the channel must not be dropped.
    pub unsafe fn get_async<B: ServerRef>(
        &mut self,
        key: &str,
        token: &RdmaToken,
        remote_offset: u64,
        source: Source<'_, B>,
        handle: NonZeroUsize,
    ) -> Result<()> {
        let call = self.prepare(key, token, remote_offset, source)?;
        // SAFETY: as in `get` for the submission; the caller keeps the
        // buffer and channel alive until the completion arrives.
        let rc = unsafe {
            sys::cuobj_get_async(
                self.server.as_ptr(),
                call.key.as_ptr(),
                call.buffer.as_ptr(),
                call.local_offset,
                call.remote_addr,
                call.len,
                call.descriptor.as_ptr(),
                self.id,
                handle.get() as *mut std::ffi::c_void,
            )
        };
        transfer_result(rc, 0).map(drop)
    }

    /// The channel's completion descriptor: readable while completions are
    /// pending. It is non-blocking, owned by the library and never closed by
    /// this crate; register it with a reactor (for example
    /// `tokio::io::unix::AsyncFd::with_interest(fd, Interest::READABLE)`).
    /// It holds the server rather than the channel, so it can stay registered
    /// while [`process_completions`](Self::process_completions) drains it.
    pub fn completion_fd(&self) -> Result<CompletionFd<S>> {
        Ok(CompletionFd {
            fd: self.raw_completion_fd()?,
            _server: self.server.clone(),
        })
    }

    /// Drains completions into `out` and returns how many were filled. When
    /// it fills all of `out`, call again before waiting on the descriptor:
    /// the remaining completions raise no new notification.
    pub fn process_completions(&mut self, out: &mut [Completion]) -> Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        let fd = self.raw_completion_fd()?;
        let max = c_int::try_from(out.len()).unwrap_or(c_int::MAX);
        // SAFETY: the server is live; `Completion` has the wrapper's layout
        // (asserted below) and `out` holds at least `max` entries.
        let rc = unsafe {
            sys::cuobj_process_completions(self.server.as_ptr(), fd, out.as_mut_ptr().cast(), max)
        };
        if rc < 0 {
            return Err(library_error("process completions", rc as isize));
        }
        Ok(rc as usize)
    }

    fn raw_completion_fd(&self) -> Result<RawFd> {
        // SAFETY: the server is live.
        let fd = unsafe { sys::cuobj_channel_completion_fd(self.server.as_ptr(), self.id) };
        if fd < 0 {
            return Err(Error::AsyncUnavailable(self.id));
        }
        Ok(fd)
    }

    /// Checks a transfer against both buffers and builds its arguments.
    fn prepare<B: ServerRef>(
        &self,
        key: &str,
        token: &RdmaToken,
        remote_offset: u64,
        source: Source<'_, B>,
    ) -> Result<Call> {
        let (server, buffer, local_offset, len) = match source {
            Source::Range {
                buffer,
                offset,
                len,
            } => {
                buffer.check(offset, len)?;
                (&*buffer.server, buffer.raw, offset as u64, len)
            }
            // The library ignores the local offset of a list handle.
            Source::List(list) => (&*list.base.server, list.raw, 0, list.len()),
        };
        if len == 0 {
            return Err(Error::InvalidArgument("zero-length transfer"));
        }
        if !ptr::eq(server, &*self.server) {
            return Err(Error::InvalidArgument(
                "buffer is registered with another server",
            ));
        }
        let needed = remote_offset
            .checked_add(len as u64)
            .ok_or(Error::InvalidArgument("remote range overflows"))?;
        if needed > token.buffer_len() {
            return Err(Error::TokenTooSmall {
                buffer: token.buffer_len(),
                needed,
            });
        }
        Ok(Call {
            key: CString::new(key).map_err(|_| Error::InvalidArgument("key contains NUL"))?,
            descriptor: CString::new(token.as_str())
                .map_err(|_| Error::InvalidToken("contains NUL"))?,
            buffer,
            local_offset,
            remote_addr: token
                .remote_addr()
                .checked_add(remote_offset)
                .ok_or(Error::InvalidArgument("remote address overflows"))?,
            len,
        })
    }
}

impl<S: ServerRef> Drop for Channel<S> {
    fn drop(&mut self) {
        // SAFETY: the server is live and the channel came from it.
        unsafe { sys::cuobj_channel_free(self.server.as_ptr(), self.id) };
    }
}

impl<S: ServerRef> fmt::Debug for Channel<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Channel").field("id", &self.id).finish()
    }
}

struct Call {
    key: CString,
    descriptor: CString,
    buffer: NonNull<sys::cuobj_buffer>,
    local_offset: u64,
    remote_addr: u64,
    len: usize,
}

fn transfer_result(rc: isize, verbs_status: c_int) -> Result<usize> {
    if rc < 0 {
        return Err(Error::Transfer {
            verbs_status,
            source: errno(rc),
        });
    }
    Ok(rc as usize)
}

/// A channel's completion descriptor. It keeps the server alive and is never
/// closed by this crate.
pub struct CompletionFd<S: ServerRef> {
    fd: RawFd,
    _server: S,
}

impl<S: ServerRef> AsRawFd for CompletionFd<S> {
    fn as_raw_fd(&self) -> RawFd {
        self.fd
    }
}

impl<S: ServerRef> fmt::Debug for CompletionFd<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompletionFd")
            .field("fd", &self.fd)
            .finish()
    }
}

/// One finished asynchronous transfer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Completion {
    handle: usize,
    status: isize,
}

const _: () = assert!(
    size_of::<Completion>() == size_of::<sys::cuobj_completion>()
        && align_of::<Completion>() == align_of::<sys::cuobj_completion>()
        && offset_of!(Completion, handle) == offset_of!(sys::cuobj_completion, handle)
        && offset_of!(Completion, status) == offset_of!(sys::cuobj_completion, status)
);

impl Completion {
    /// The handle given to [`Channel::get_async`].
    pub fn handle(&self) -> usize {
        self.handle
    }

    /// The transfer's outcome. The byte count is only reported by the
    /// blocking calls; a successful asynchronous GET moved its whole range.
    pub fn result(&self) -> io::Result<()> {
        if self.status < 0 {
            return Err(errno(self.status));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_status_maps_to_errno() {
        assert!(
            Completion {
                handle: 7,
                status: 0
            }
            .result()
            .is_ok()
        );
        let failed = Completion {
            handle: 7,
            status: -5,
        };
        assert_eq!(failed.handle(), 7);
        assert_eq!(failed.result().unwrap_err().raw_os_error(), Some(5));
    }

    #[test]
    fn borrowed_and_arc_handles_cross_threads() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<Server>();
        send_sync::<RegisteredBuffer<&Server>>();
        send_sync::<RegisteredBuffer<Arc<Server>>>();
        send_sync::<SgBuffer<'_, Arc<Server>>>();
        send_sync::<Channel<&Server>>();
        send_sync::<Channel<Arc<Server>>>();
        send_sync::<CompletionFd<Arc<Server>>>();
    }
}
