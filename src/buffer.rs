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

//! Registered host memory and scatter-gather lists over it.

use crate::server::ServerRef;
use crate::{Error, Result, library_error};
use cuobject_sys as sys;
use memmap2::{MmapMut, MmapRaw};
use std::fmt;
use std::mem::{align_of, offset_of, size_of};
use std::ptr::{self, NonNull};

/// Page-aligned host memory registered with a [`Server`](crate::Server) for
/// RDMA, deregistered and unmapped on drop.
///
/// `S` is how the buffer holds its server ([`ServerRef`]): `&Server` borrows
/// it, `Arc<Server>` or `Rc<Server>` share ownership of it. To share the
/// buffer itself, borrow it or wrap it in an `Arc`.
pub struct RegisteredBuffer<S: ServerRef> {
    pub(crate) server: S,
    pub(crate) raw: NonNull<sys::cuobj_buffer>,
    // Unmapped after `Drop` deregistered it.
    map: MmapRaw,
}

// SAFETY: the handle is only passed back to the library, which serializes
// registrations internally; Rust access to the memory goes through the
// `unsafe` accessors, whose callers rule out data races. The server holder
// moves or is shared along with the buffer, hence the bounds on `S`.
unsafe impl<S: ServerRef + Send> Send for RegisteredBuffer<S> {}
unsafe impl<S: ServerRef + Sync> Sync for RegisteredBuffer<S> {}

impl<S: ServerRef> Drop for RegisteredBuffer<S> {
    fn drop(&mut self) {
        // SAFETY: the handle came from `register` on this server, which
        // `server` keeps alive; every scatter-gather list over it borrows this
        // buffer, so none is left.
        unsafe { sys::cuobj_deregister(self.server.as_ptr(), self.raw.as_ptr()) };
    }
}

impl<S: ServerRef> RegisteredBuffer<S> {
    /// Maps `len` bytes of zero-filled, page-aligned host memory and
    /// registers it with `server` for RDMA.
    /// [`Server::register_host_buffer`](crate::Server::register_host_buffer)
    /// is the shorthand for a borrowed server.
    pub fn new(server: S, len: usize) -> Result<Self> {
        if len == 0 {
            return Err(Error::InvalidArgument("zero-length buffer"));
        }
        let map: MmapRaw = MmapMut::map_anon(len)
            .map_err(|source| Error::Library {
                op: "map host buffer",
                source,
            })?
            .into();
        let mut out = ptr::null_mut();
        // SAFETY: the server is live; `map` is `len` writable bytes that stay
        // mapped until after the registration is released.
        let rc =
            unsafe { sys::cuobj_register(server.as_ptr(), map.as_mut_ptr().cast(), len, &mut out) };
        if rc < 0 {
            return Err(library_error("register buffer", rc as isize));
        }
        let raw =
            NonNull::new(out).ok_or(Error::InvalidArgument("library returned no buffer handle"))?;
        Ok(Self { server, raw, map })
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the buffer is empty (never true: registration needs a length).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Start of the memory. Dereferencing it carries the obligations of
    /// [`slice_mut`](Self::slice_mut).
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.map.as_mut_ptr()
    }

    pub(crate) fn check(&self, offset: usize, len: usize) -> Result<()> {
        if offset.checked_add(len).is_none_or(|end| end > self.len()) {
            return Err(Error::InvalidArgument(
                "range outside the registered buffer",
            ));
        }
        Ok(())
    }

    /// Borrows `len` bytes at `offset`.
    ///
    /// # Safety
    ///
    /// While the slice lives, no [`Channel::put`](crate::Channel::put) may
    /// write into the range and no slice from
    /// [`slice_mut`](Self::slice_mut) may overlap it.
    pub unsafe fn slice(&self, offset: usize, len: usize) -> Result<&[u8]> {
        self.check(offset, len)?;
        // SAFETY: the range is inside the mapping; the caller rules out
        // concurrent writers.
        Ok(unsafe { std::slice::from_raw_parts(self.map.as_ptr().add(offset), len) })
    }

    /// Mutably borrows `len` bytes at `offset`.
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
        self.check(offset, len)?;
        // SAFETY: as above; the caller holds the only access to the range.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.map.as_mut_ptr().add(offset), len) })
    }

    /// Builds a scatter-gather list over `entries` of this buffer; one GET
    /// then moves all ranges, in order, as one RDMA operation. The library
    /// accepts at most the server's `max_sge` entries
    /// ([`DEFAULT_MAX_SG_ENTRIES`](crate::DEFAULT_MAX_SG_ENTRIES) by default).
    pub fn sg_buffer(&self, entries: &[SgEntry]) -> Result<SgBuffer<'_, S>> {
        if entries.is_empty() {
            return Err(Error::InvalidArgument("empty scatter-gather list"));
        }
        let mut total = 0usize;
        for entry in entries {
            if entry
                .offset
                .checked_add(entry.size as u64)
                .is_none_or(|end| end > self.len() as u64)
            {
                return Err(Error::InvalidArgument(
                    "scatter-gather entry outside the registered buffer",
                ));
            }
            total = total
                .checked_add(entry.size)
                .ok_or(Error::InvalidArgument("scatter-gather length overflows"))?;
        }
        let mut out = ptr::null_mut();
        // SAFETY: the server and registration are live; `SgEntry` has the
        // wrapper's entry layout (asserted below) and `entries` is valid for
        // its length.
        let rc = unsafe {
            sys::cuobj_sg_buffer(
                self.server.as_ptr(),
                self.raw.as_ptr(),
                entries.as_ptr().cast(),
                entries.len(),
                &mut out,
            )
        };
        if rc < 0 {
            return Err(library_error("build scatter-gather list", rc as isize));
        }
        let raw =
            NonNull::new(out).ok_or(Error::InvalidArgument("library returned no list handle"))?;
        Ok(SgBuffer {
            base: self,
            raw,
            len: total,
        })
    }
}

impl<S: ServerRef> fmt::Debug for RegisteredBuffer<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegisteredBuffer")
            .field("len", &self.len())
            .finish()
    }
}

/// One `(offset, size)` range of a [`RegisteredBuffer`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SgEntry {
    /// Byte offset in the buffer.
    pub offset: u64,
    /// Length in bytes.
    pub size: usize,
}

const _: () = assert!(
    size_of::<SgEntry>() == size_of::<sys::cuobj_sg_entry>()
        && align_of::<SgEntry>() == align_of::<sys::cuobj_sg_entry>()
        && offset_of!(SgEntry, offset) == offset_of!(sys::cuobj_sg_entry, offset)
        && offset_of!(SgEntry, size) == offset_of!(sys::cuobj_sg_entry, size)
);

/// A scatter-gather list over ranges of one [`RegisteredBuffer`], released on
/// drop. It borrows the buffer, which therefore stays registered.
pub struct SgBuffer<'b, S: ServerRef> {
    pub(crate) base: &'b RegisteredBuffer<S>,
    pub(crate) raw: NonNull<sys::cuobj_buffer>,
    len: usize,
}

// SAFETY: as for `RegisteredBuffer`: the handle is only passed back to the
// library. The list holds `&RegisteredBuffer<S>`, which crosses threads when
// the buffer is `Sync`.
unsafe impl<S: ServerRef + Sync> Send for SgBuffer<'_, S> {}
unsafe impl<S: ServerRef + Sync> Sync for SgBuffer<'_, S> {}

impl<S: ServerRef> SgBuffer<'_, S> {
    /// Total bytes over all ranges.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the list moves no bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<S: ServerRef> Drop for SgBuffer<'_, S> {
    fn drop(&mut self) {
        // SAFETY: the handle came from `sg_buffer` on the borrowed buffer,
        // which is still registered and keeps its server alive.
        unsafe { sys::cuobj_sg_buffer_put(self.base.server.as_ptr(), self.raw.as_ptr()) };
    }
}

impl<S: ServerRef> fmt::Debug for SgBuffer<'_, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SgBuffer").field("len", &self.len).finish()
    }
}
