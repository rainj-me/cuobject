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

//! The client: registered client memory and the transfers the library drives
//! through a callback.

use crate::{EINVAL, EIO, EPROTO, Error, Result, library_error};
use cuobject_sys as sys;
use memmap2::{MmapMut, MmapRaw};
use std::ffi::{CStr, c_char, c_void};
use std::fmt;
use std::io;
use std::ops::Deref;
use std::panic::{self, AssertUnwindSafe};
use std::ptr::{self, NonNull};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Version of `cuobjclient.h` this crate was compiled against, as
/// `(major, minor)`.
pub fn client_version() -> (u32, u32) {
    // SAFETY: a pure function of compile-time constants.
    let packed = unsafe { sys::cuobj_client_version() };
    let packed = u32::try_from(packed).unwrap_or(0);
    (packed / 1000, packed % 1000)
}

/// A cuObject client on the RDMA DC protocol, destroyed on drop.
///
/// The library picks its RDMA devices from the cuFile configuration
/// (`cufile.json`, located through `CUFILE_ENV_PATH_JSON`) and registers
/// memory through cuFile's user-space RDMA, which needs the `nvidia_peermem`
/// kernel module or, without it, `CUFILE_DMABUF_ENABLE=true`; otherwise
/// registration fails with `EIO`. Every buffer
/// registered with it holds it through a [`ClientRef`], so the client
/// outlives them all.
///
/// # Example
///
/// ```no_run
/// use cuobject::{Client, TOKEN_HEADER};
///
/// # fn send(token: &str, offset: u64, len: usize) -> std::io::Result<usize> { Ok(len) }
/// # fn main() -> cuobject::Result<()> {
/// let client = Client::new()?;
/// let buffer = client.register_host_buffer(1 << 20)?;
/// // The library calls the closure for each chunk; it sends the S3 request
/// // with `TOKEN_HEADER` set to the chunk's token and returns the bytes the
/// // server moved.
/// let moved = buffer.get(0, buffer.len(), |chunk| send(chunk.token(), chunk.offset(), chunk.len()))?;
/// // SAFETY: the transfer finished and nothing else borrows the buffer.
/// let object = unsafe { buffer.slice(0, moved)? };
/// # Ok(())
/// # }
/// ```
pub struct Client {
    raw: NonNull<sys::cuobj_client>,
}

// SAFETY: the library serves GETs and PUTs from several callers at once (its
// callbacks may also run on other threads); registrations are serialized
// inside it.
unsafe impl Send for Client {}
unsafe impl Sync for Client {}

impl Drop for Client {
    fn drop(&mut self) {
        // SAFETY: every buffer holds this client through a `ClientRef`, so
        // none is left.
        unsafe { sys::cuobj_client_destroy(self.raw.as_ptr()) };
    }
}

impl Client {
    /// Creates a client.
    pub fn new() -> Result<Self> {
        let mut out = ptr::null_mut();
        // SAFETY: `out` is a valid out-pointer.
        let rc = unsafe { sys::cuobj_client_create(&mut out) };
        if rc < 0 {
            return Err(library_error("create client", rc as isize));
        }
        let raw =
            NonNull::new(out).ok_or(Error::InvalidArgument("library returned no client handle"))?;
        Ok(Self { raw })
    }

    /// Whether the library reports the client as connected.
    pub fn is_connected(&self) -> bool {
        // SAFETY: the client is live.
        unsafe { sys::cuobj_client_connected(self.raw.as_ptr()) == 1 }
    }

    /// [`ClientBuffer::new`] on a borrow of this client.
    pub fn register_host_buffer(&self, len: usize) -> Result<ClientBuffer<&Self>> {
        ClientBuffer::new(self, len)
    }

    /// The kind of memory at `ptr`, or `None` when the library rejects it.
    pub fn memory_type(ptr: *const u8) -> Option<MemoryType> {
        // SAFETY: the library only classifies the address.
        match unsafe { sys::cuobj_client_memory_type(ptr.cast()) } {
            sys::cuobj_client_memory::CUOBJ_CLIENT_MEMORY_SYSTEM => Some(MemoryType::System),
            sys::cuobj_client_memory::CUOBJ_CLIENT_MEMORY_CUDA_MANAGED => {
                Some(MemoryType::CudaManaged)
            }
            sys::cuobj_client_memory::CUOBJ_CLIENT_MEMORY_CUDA_DEVICE => {
                Some(MemoryType::CudaDevice)
            }
            sys::cuobj_client_memory::CUOBJ_CLIENT_MEMORY_UNKNOWN => Some(MemoryType::Unknown),
            sys::cuobj_client_memory::CUOBJ_CLIENT_MEMORY_INVALID => None,
        }
    }

    pub(crate) fn as_ptr(&self) -> *mut sys::cuobj_client {
        self.raw.as_ptr()
    }
}

impl fmt::Debug for Client {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Client").finish_non_exhaustive()
    }
}

/// The kind of memory behind a pointer, as the library classifies it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MemoryType {
    /// Host memory.
    System,
    /// CUDA managed memory.
    CudaManaged,
    /// CUDA device memory.
    CudaDevice,
    /// Memory the library could not classify.
    Unknown,
}

/// How a [`ClientBuffer`] holds the [`Client`] it was registered with:
/// `&Client` borrows it, `Arc<Client>` or `Rc<Client>` share ownership of it.
///
/// Sealed: buffers rely on every `deref` returning the same client.
pub trait ClientRef: Deref<Target = Client> + Clone + sealed::Sealed {}

impl ClientRef for &Client {}
impl ClientRef for Arc<Client> {}
impl ClientRef for Rc<Client> {}

mod sealed {
    use super::Client;
    use std::rc::Rc;
    use std::sync::Arc;

    pub trait Sealed {}

    impl Sealed for &Client {}
    impl Sealed for Arc<Client> {}
    impl Sealed for Rc<Client> {}
}

/// Client memory registered for RDMA, deregistered on drop (and unmapped,
/// when the crate mapped it).
pub struct ClientBuffer<C: ClientRef> {
    client: C,
    ptr: NonNull<u8>,
    len: usize,
    // Host memory mapped by `new`; unmapped after `Drop` deregistered it.
    // `None` for memory registered with `from_raw`, which the caller owns.
    map: Option<MmapRaw>,
}

// SAFETY: the registration is only passed back to the library; Rust access to
// the memory goes through the `unsafe` accessors, whose callers rule out data
// races. The client holder moves or is shared along with the buffer.
unsafe impl<C: ClientRef + Send> Send for ClientBuffer<C> {}
unsafe impl<C: ClientRef + Sync> Sync for ClientBuffer<C> {}

impl<C: ClientRef> Drop for ClientBuffer<C> {
    fn drop(&mut self) {
        // SAFETY: the memory was registered with this client, which `client`
        // keeps alive; every token over it borrows this buffer, so none is
        // left. A failure leaves nothing to clean up.
        unsafe { sys::cuobj_client_deregister(self.client.as_ptr(), self.ptr.as_ptr().cast()) };
    }
}

impl<C: ClientRef> ClientBuffer<C> {
    /// Maps `len` bytes of zero-filled, page-aligned host memory and
    /// registers it with `client`. [`Client::register_host_buffer`] is the
    /// shorthand for a borrowed client.
    pub fn new(client: C, len: usize) -> Result<Self> {
        if len == 0 {
            return Err(Error::InvalidArgument("zero-length buffer"));
        }
        let map: MmapRaw = MmapMut::map_anon(len)
            .map_err(|source| Error::Library {
                op: "map host buffer",
                source,
            })?
            .into();
        let ptr = NonNull::new(map.as_mut_ptr())
            .ok_or(Error::InvalidArgument("mapping returned a null pointer"))?;
        // SAFETY: `map` is `len` writable bytes that stay mapped until after
        // the registration is released.
        unsafe { Self::register(client, ptr, len, Some(map)) }
    }

    /// Registers `len` bytes at `ptr`, host or CUDA memory the caller owns.
    ///
    /// # Safety
    ///
    /// The memory must stay allocated until the buffer is dropped, and no Rust
    /// reference may access it while a transfer over it runs.
    pub unsafe fn from_raw(client: C, ptr: NonNull<u8>, len: usize) -> Result<Self> {
        if len == 0 {
            return Err(Error::InvalidArgument("zero-length buffer"));
        }
        // SAFETY: the caller keeps the memory allocated for the buffer's life.
        unsafe { Self::register(client, ptr, len, None) }
    }

    /// # Safety
    ///
    /// `ptr` is `len` bytes that stay allocated until the buffer is dropped.
    unsafe fn register(
        client: C,
        ptr: NonNull<u8>,
        len: usize,
        map: Option<MmapRaw>,
    ) -> Result<Self> {
        // SAFETY: the client is live; the caller vouches for the memory.
        let rc = unsafe { sys::cuobj_client_register(client.as_ptr(), ptr.as_ptr().cast(), len) };
        if rc < 0 {
            return Err(library_error("register client buffer", rc as isize));
        }
        Ok(Self {
            client,
            ptr,
            len,
            map,
        })
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty (never true: registration needs a length).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Start of the memory. Dereferencing it carries the obligations of
    /// [`slice_mut`](Self::slice_mut).
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// The largest chunk one callback moves: a transfer longer than this
    /// calls the closure several times.
    pub fn max_request_size(&self) -> Result<usize> {
        // SAFETY: the client and registration are live.
        let rc = unsafe {
            sys::cuobj_client_max_request_size(self.client.as_ptr(), self.ptr.as_ptr().cast())
        };
        if rc < 0 {
            return Err(library_error("query max request size", rc));
        }
        Ok(rc as usize)
    }

    fn check(&self, offset: usize, len: usize) -> Result<()> {
        if len == 0 {
            return Err(Error::InvalidArgument("zero-length range"));
        }
        if offset.checked_add(len).is_none_or(|end| end > self.len) {
            return Err(Error::InvalidArgument(
                "range outside the registered buffer",
            ));
        }
        Ok(())
    }

    fn host(&self, offset: usize, len: usize) -> Result<*mut u8> {
        if self.map.is_none() {
            return Err(Error::InvalidArgument(
                "only host memory mapped by ClientBuffer::new can be sliced",
            ));
        }
        self.check(offset, len)?;
        // SAFETY: the range is inside the mapping.
        Ok(unsafe { self.ptr.as_ptr().add(offset) })
    }

    /// Borrows `len` bytes at `offset`. Fails for memory registered with
    /// [`from_raw`](Self::from_raw), which may not be host memory.
    ///
    /// # Safety
    ///
    /// While the slice lives, no GET may write into the range and no slice
    /// from [`slice_mut`](Self::slice_mut) may overlap it.
    pub unsafe fn slice(&self, offset: usize, len: usize) -> Result<&[u8]> {
        let ptr = self.host(offset, len)?;
        // SAFETY: the range is mapped host memory; the caller rules out
        // concurrent writers.
        Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// Mutably borrows `len` bytes at `offset`. Fails for memory registered
    /// with [`from_raw`](Self::from_raw), which may not be host memory.
    ///
    /// # Safety
    ///
    /// While the slice lives, no transfer may read or write the range and no
    /// other slice from [`slice`](Self::slice) or `slice_mut` may overlap it.
    #[allow(
        clippy::mut_from_ref,
        reason = "exclusivity is the caller's obligation, as for a raw pointer"
    )]
    pub unsafe fn slice_mut(&self, offset: usize, len: usize) -> Result<&mut [u8]> {
        let ptr = self.host(offset, len)?;
        // SAFETY: as above; the caller holds the only access to the range.
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
    }

    /// GETs `len` bytes of an object into the buffer at `offset` and returns
    /// the bytes moved.
    ///
    /// The library calls `io` for each chunk, possibly from other threads and
    /// at the same time; `io` sends the request with the chunk's token and
    /// returns the bytes the server moved, or an error (its errno reaches the
    /// library). A panic in `io` fails the chunk; once the transfer returns,
    /// this call panics in turn (the original message was already reported
    /// by the panic hook).
    ///
    /// An `EIO` failure after the library reset its RDMA token means the
    /// transfer can be retried.
    pub fn get<F>(&self, offset: usize, len: usize, io: F) -> Result<usize>
    where
        F: Fn(Chunk<'_>) -> io::Result<usize> + Sync,
    {
        self.transfer(Operation::Get, offset, len, io)
    }

    /// PUTs `len` bytes at `offset` of the buffer as an object and returns
    /// the bytes moved; `io` is called as for [`get`](Self::get).
    pub fn put<F>(&self, offset: usize, len: usize, io: F) -> Result<usize>
    where
        F: Fn(Chunk<'_>) -> io::Result<usize> + Sync,
    {
        self.transfer(Operation::Put, offset, len, io)
    }

    fn transfer<F>(&self, op: Operation, offset: usize, len: usize, io: F) -> Result<usize>
    where
        F: Fn(Chunk<'_>) -> io::Result<usize> + Sync,
    {
        self.check(offset, len)?;
        let call = Call {
            io: &io,
            panicked: AtomicBool::new(false),
        };
        let (run, name) = match op {
            Operation::Get => (sys::cuobj_client_get as Run, "client GET"),
            Operation::Put => (sys::cuobj_client_put as Run, "client PUT"),
        };
        // SAFETY: the client and registration are live; the range lies inside
        // the registration; `call` outlives the synchronous library call, and
        // `trampoline::<F>` reads it as the `Call<F>` it is.
        let rc = unsafe {
            run(
                self.client.as_ptr(),
                self.ptr.as_ptr().add(offset).cast(),
                len,
                Some(trampoline::<F>),
                ptr::from_ref(&call).cast_mut().cast(),
            )
        };
        if call.panicked.into_inner() {
            panic!("a cuObject {name} chunk callback panicked");
        }
        if rc < 0 {
            return Err(library_error(name, rc));
        }
        Ok(rc as usize)
    }

    /// Creates a token for `len` bytes at `offset`, for a transfer driven
    /// without [`get`](Self::get) or [`put`](Self::put): send it as the
    /// [`TOKEN_HEADER`](crate::TOKEN_HEADER) value. The token is released on
    /// drop and borrows the buffer, which stays registered meanwhile.
    pub fn token(&self, offset: usize, len: usize, op: Operation) -> Result<ClientToken<'_, C>> {
        self.check(offset, len)?;
        let op = match op {
            Operation::Get => sys::cuobj_client_op::CUOBJ_CLIENT_GET,
            Operation::Put => sys::cuobj_client_op::CUOBJ_CLIENT_PUT,
        };
        let mut out = ptr::null_mut();
        // SAFETY: the client and registration are live, the range lies inside
        // the registration and `out` is a valid out-pointer.
        let rc = unsafe {
            sys::cuobj_client_token(
                self.client.as_ptr(),
                self.ptr.as_ptr().cast(),
                len,
                offset,
                op,
                &mut out,
            )
        };
        if rc < 0 {
            return Err(library_error("create token", rc as isize));
        }
        let raw = NonNull::new(out).ok_or(Error::InvalidArgument("library returned no token"))?;
        let token = ClientToken { buffer: self, raw };
        // SAFETY: the library returns a NUL-terminated string that lives until
        // `cuobj_client_token_put`.
        unsafe { CStr::from_ptr(raw.as_ptr()) }
            .to_str()
            .map_err(|_| Error::InvalidToken("not UTF-8"))?;
        Ok(token)
    }
}

impl<C: ClientRef> fmt::Debug for ClientBuffer<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClientBuffer")
            .field("len", &self.len)
            .field("host_mapped", &self.map.is_some())
            .finish()
    }
}

/// The direction of a transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    /// The server writes into client memory.
    Get,
    /// The server reads from client memory.
    Put,
}

/// One chunk of a client transfer, handed to the closure of
/// [`ClientBuffer::get`] or [`ClientBuffer::put`].
#[derive(Debug, Clone, Copy)]
pub struct Chunk<'a> {
    token: &'a str,
    offset: u64,
    len: usize,
}

impl<'a> Chunk<'a> {
    /// The [`TOKEN_HEADER`](crate::TOKEN_HEADER) value describing the chunk's
    /// memory.
    pub fn token(&self) -> &'a str {
        self.token
    }

    /// Where the chunk starts in the object.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Length of the chunk in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the chunk is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A token from [`ClientBuffer::token`], released on drop.
pub struct ClientToken<'b, C: ClientRef> {
    buffer: &'b ClientBuffer<C>,
    raw: NonNull<c_char>,
}

// SAFETY: the token is a string owned by the library until it is released;
// it holds `&ClientBuffer<C>`, which crosses threads when the buffer is `Sync`.
unsafe impl<C: ClientRef + Sync> Send for ClientToken<'_, C> {}
unsafe impl<C: ClientRef + Sync> Sync for ClientToken<'_, C> {}

impl<C: ClientRef> ClientToken<'_, C> {
    /// The token, for the [`TOKEN_HEADER`](crate::TOKEN_HEADER) header.
    pub fn as_str(&self) -> &str {
        // SAFETY: NUL-terminated and alive until drop; checked to be UTF-8
        // when the token was created.
        let bytes = unsafe { CStr::from_ptr(self.raw.as_ptr()) }.to_bytes();
        // SAFETY: as above.
        unsafe { std::str::from_utf8_unchecked(bytes) }
    }
}

impl<C: ClientRef> Drop for ClientToken<'_, C> {
    fn drop(&mut self) {
        // SAFETY: the token came from this buffer's client, and the borrowed
        // buffer is still registered.
        unsafe { sys::cuobj_client_token_put(self.buffer.client.as_ptr(), self.raw.as_ptr()) };
    }
}

impl<C: ClientRef> fmt::Debug for ClientToken<'_, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ClientToken").field(&self.as_str()).finish()
    }
}

type Run = unsafe extern "C" fn(
    *mut sys::cuobj_client,
    *mut c_void,
    usize,
    sys::cuobj_client_io_fn,
    *mut c_void,
) -> isize;

/// The state behind the library's `user` pointer during one transfer.
struct Call<'f, F> {
    io: &'f F,
    // Set when `io` panicked on any chunk. Callbacks may run concurrently, so
    // it is atomic; it is read only after the library call returned, when no
    // callback runs any more, so `Relaxed` suffices.
    panicked: AtomicBool,
}

/// The library's chunk callback: calls the transfer's closure.
///
/// # Safety
///
/// `user` is the `Call<F>` of a running transfer and `token` is `token_len`
/// readable bytes.
unsafe extern "C" fn trampoline<F>(
    user: *mut c_void,
    _ptr: *mut c_char,
    size: usize,
    offset: i64,
    token: *const c_char,
    token_len: usize,
) -> isize
where
    F: Fn(Chunk<'_>) -> io::Result<usize> + Sync,
{
    // SAFETY: the caller passes the transfer's `Call<F>`, shared across the
    // library's threads as `F: Sync` allows.
    let call = unsafe { &*user.cast_const().cast::<Call<'_, F>>() };
    if token.is_null() {
        return -(EINVAL as isize);
    }
    // SAFETY: the caller passes `token_len` readable bytes.
    let token = unsafe { std::slice::from_raw_parts(token.cast::<u8>(), token_len) };
    let Ok(token) = std::str::from_utf8(token) else {
        return -(EPROTO as isize);
    };
    let Ok(offset) = u64::try_from(offset) else {
        return -(EINVAL as isize);
    };
    let chunk = Chunk {
        token,
        offset,
        len: size,
    };
    match panic::catch_unwind(AssertUnwindSafe(|| (call.io)(chunk))) {
        Ok(Ok(moved)) => isize::try_from(moved).unwrap_or(-(EINVAL as isize)),
        Ok(Err(err)) => -(err.raw_os_error().filter(|&code| code > 0).unwrap_or(EIO) as isize),
        Err(_) => {
            call.panicked.store(true, Ordering::Relaxed);
            -(EIO as isize)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Calls the trampoline as the library would and reports its return and
    /// whether the closure panicked.
    fn run<F>(io: F, token: &str, offset: i64) -> (isize, bool)
    where
        F: Fn(Chunk<'_>) -> io::Result<usize> + Sync,
    {
        let call = Call {
            io: &io,
            panicked: AtomicBool::new(false),
        };
        // SAFETY: `call` is a live `Call<F>` and `token` is readable.
        let rc = unsafe {
            trampoline::<F>(
                ptr::from_ref(&call).cast_mut().cast(),
                ptr::null_mut(),
                4096,
                offset,
                token.as_ptr().cast(),
                token.len(),
            )
        };
        (rc, call.panicked.into_inner())
    }

    #[test]
    fn trampoline_maps_closure_results() {
        let seen = |chunk: Chunk<'_>| {
            assert_eq!(
                (chunk.token(), chunk.offset(), chunk.len()),
                ("a:b", 8, 4096)
            );
            Ok(chunk.len())
        };
        assert_eq!(run(seen, "a:b", 8), (4096, false));
        assert_eq!(
            run(|_| Err(io::Error::from_raw_os_error(110)), "t", 0),
            (-110, false)
        );
        assert_eq!(
            run(|_| Err(io::Error::other("no errno")), "t", 0),
            (-(EIO as isize), false)
        );
        assert_eq!(run(|_| Ok(1), "t", -1), (-(EINVAL as isize), false));
    }

    #[test]
    fn trampoline_catches_panics() {
        assert_eq!(run(|_| panic!("boom"), "t", 0), (-(EIO as isize), true));
    }

    #[test]
    fn handles_cross_threads() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<Client>();
        send_sync::<ClientBuffer<&Client>>();
        send_sync::<ClientBuffer<Arc<Client>>>();
        send_sync::<ClientToken<'_, Arc<Client>>>();
    }
}
