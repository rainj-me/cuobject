# cuobject

Safe Rust bindings to NVIDIA's cuObject libraries for S3-over-RDMA (GPUDirect object loading). A client sends an S3 request with an `x-amz-rdma-token` header describing a buffer in its host or GPU memory; the server moves the object bytes with RDMA instead of carrying them in the HTTP body. The crate covers both ends:

| Feature | Library | Types |
|---|---|---|
| `server` (default) | `libcuobjserver` | `Server`, `RegisteredBuffer`, `Channel` |
| `client` | `libcuobjclient`, `libcufile` | `Client`, `ClientBuffer` |

The server moves a chunk into the client memory a token describes:

```rust,no_run
use cuobject::{RdmaToken, Server, ServerOptions, Source};

# fn main() -> cuobject::Result<()> {
let server = Server::new("10.0.0.1".parse().unwrap(), cuobject::DEFAULT_PORT, &ServerOptions::default())?;
let staging = server.register_host_buffer(1 << 20)?;
// SAFETY: no transfer uses `staging` yet and nothing else borrows it.
unsafe { staging.slice_mut(0, 5)? }.copy_from_slice(b"hello");

let token = RdmaToken::parse("0000f52eddc00000:00100000:001824d2:0000:002873:1:00000000000000000000ffff0a0d5410")?;
let mut channel = server.channel()?;
channel.get("bucket/key", &token, 0, Source::Range { buffer: &staging, offset: 0, len: 5 })?;
# Ok(())
# }
```

The client registers memory and, for each chunk the library hands it, sends the token to the server:

```rust,ignore
use cuobject::Client;

let client = Client::new()?;
let buffer = client.register_host_buffer(1 << 20)?;
let moved = buffer.get(0, buffer.len(), |chunk| {
    // Send GET with `x-amz-rdma-token: chunk.token()` for `chunk.len()` bytes at
    // `chunk.offset()`; return the server's x-amz-rdma-bytes-transferred.
    send_get(chunk.token(), chunk.offset(), chunk.len())
})?;
```

## What the crate guarantees

- Server-side handles hold their server through a `ServerRef`, client buffers their client through a `ClientRef`, chosen by you: `&T` borrows it, `Arc<T>` or `Rc<T>` share ownership of it. Scatter-gather lists and client tokens borrow their buffer. Either way no handle outlives the library state it uses, and the crate itself never allocates a reference count.
- A channel is used through `&mut Channel`: the library forbids using one channel from two threads at once.
- Every transfer is checked against the registered buffer and the client token before the library sees it.
- Host buffers are anonymous mappings owned by the crate and unmapped after deregistration (the server library's own allocator has no free call).
- The client's chunk callback may run on library threads, concurrently: it is `Fn + Sync`, and a panic in it fails the chunk and makes the transfer panic on the calling thread once it returns.

Reading or writing registered memory from Rust (`slice`, `slice_mut`), registering caller-owned client memory (`ClientBuffer::from_raw`) and submitting asynchronous GETs (`get_async`) stay `unsafe`: the NIC accesses that memory outside Rust's view, so the caller rules out overlapping access.

## Examples

`examples/server.rs` is a minimal in-memory object server; `examples/client.rs` PUTs an object to it over RDMA, GETs it back and checks every byte. They talk plain HTTP/1.1 for the control path (`examples/common/http.rs`).

```sh
# On a host whose RDMA NIC owns 10.0.0.1:
cargo run --release --example server -- --rdma-addr 10.0.0.1 --listen 0.0.0.0:8080

# The client finds its RDMA NICs through cuFile's configuration: list them
# (IP addresses or device names) in `properties.rdma_dev_addr_list`.
CUFILE_DMABUF_ENABLE=true CUFILE_ENV_PATH_JSON=examples/cufile.json \
  cargo run --release --features client --example client -- --server 10.0.0.1:8080 --size-mib 16
```

cuObject sets up its RDMA connections with `rdma_cm`, so the server address must belong to an RDMA-capable interface (IPoIB or RoCE) in the process's network namespace; in Kubernetes that usually means `hostNetwork: true` or a secondary RDMA network.

The client registers memory through cuFile's user-space RDMA, which needs either the `nvidia_peermem` kernel module or DMA-BUF. With NVIDIA's open kernel modules and the kernel's own RDMA stack, `nvidia_peermem` cannot load (it needs MLNX_OFED's peer-memory API), so set `CUFILE_DMABUF_ENABLE=true` (the string `true`; `1` is ignored). Without either, cuFile logs `nvidia_peermem.ko is not loaded. Disabling UserSpace RDMA access.` and every registration fails with `EIO`.

## Requirements

- Linux with an RDMA-capable NIC, a C++17 compiler and libclang (the bindings are generated with bindgen).
- `server`: `libcuobjserver` and its headers (tested with 2.1); no particular CUDA version.
- `client`: **CUDA 13.4** — `libcuobjclient` 1.3, `libcufile` and `cuda.h` from the 13.4 toolkit (`libcuobjclient-dev-13-4`, `libcufile-dev-13-4`, `cuda-cudart-dev-13-4`). Tested on driver 580. With several toolkits installed, build with `CUDA_HOME=/usr/local/cuda-13.4`.
- At run time the client also needs `nvidia_peermem` or `CUFILE_DMABUF_ENABLE=true` (see [Examples](#examples)).

See [`cuobject-sys`](https://crates.io/crates/cuobject-sys) for the install command and how the libraries are found.

## Tests

Unit tests need only the libraries to link. The device test needs an RDMA NIC:

```sh
CUOBJECT_TEST_ADDR=<address of an RDMA NIC> cargo test -- --ignored
```

## License

Apache-2.0. `libcuobjserver`, `libcuobjclient` and `libcufile` are NVIDIA software under NVIDIA's license; this crate neither bundles nor redistributes them.
