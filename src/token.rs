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

//! The S3-over-RDMA wire protocol: headers, limits and the client token.

use crate::{Error, Result};

/// Request header carrying the client's RDMA token.
pub const TOKEN_HEADER: &str = "x-amz-rdma-token";
/// Response header carrying the transfer status as an HTTP-style code.
pub const REPLY_HEADER: &str = "x-amz-rdma-reply";
/// Response header carrying the payload bytes moved over RDMA.
pub const BYTES_TRANSFERRED_HEADER: &str = "x-amz-rdma-bytes-transferred";
/// [`REPLY_HEADER`] code for a declined transfer; the client then fails or
/// falls back to HTTP.
pub const REPLY_NOT_IMPLEMENTED: u16 = 501;
/// Largest single GET the library accepts.
pub const MAX_TRANSFER_BYTES: usize = 1 << 30;
/// The library's default connection setup port.
pub const DEFAULT_PORT: u16 = 18515;
/// The library's default limit on scatter-gather entries per list.
pub const DEFAULT_MAX_SG_ENTRIES: usize = 10;

/// A parsed [`TOKEN_HEADER`] value: colon-separated hex fields
/// `addr:size:rkey:lid:dctn:gid_present:gid`. Only the address and size are
/// read here; the library consumes the rest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdmaToken {
    raw: String,
    remote_addr: u64,
    buffer_len: u64,
}

impl RdmaToken {
    /// Parses a header value; surrounding whitespace is ignored.
    pub fn parse(raw: &str) -> Result<Self> {
        let raw = raw.trim();
        if raw.contains('\0') {
            return Err(Error::InvalidToken("contains NUL"));
        }
        let mut fields = raw.split(':');
        let addr = fields
            .next()
            .ok_or(Error::InvalidToken("missing address field"))?;
        let size = fields
            .next()
            .ok_or(Error::InvalidToken("missing size field"))?;
        if fields.count() < 4 {
            return Err(Error::InvalidToken("fewer than six fields"));
        }
        let remote_addr =
            u64::from_str_radix(addr, 16).map_err(|_| Error::InvalidToken("address is not hex"))?;
        let buffer_len =
            u64::from_str_radix(size, 16).map_err(|_| Error::InvalidToken("size is not hex"))?;
        if buffer_len == 0 {
            return Err(Error::InvalidToken("zero-length buffer"));
        }
        Ok(Self {
            raw: raw.to_owned(),
            remote_addr,
            buffer_len,
        })
    }

    /// The token as received (trimmed); the library's descriptor.
    pub fn as_str(&self) -> &str {
        &self.raw
    }

    /// Start of the client buffer.
    pub fn remote_addr(&self) -> u64 {
        self.remote_addr
    }

    /// Length of the client buffer in bytes.
    pub fn buffer_len(&self) -> u64 {
        self.buffer_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKEN: &str =
        "0000f52eddc00000:00100000:001824d2:0000:002873:1:00000000000000000000ffff0a0d5410";

    #[test]
    fn token_reads_address_and_size() {
        let token = RdmaToken::parse(&format!(" {TOKEN}\n")).expect("token");
        assert_eq!(token.remote_addr(), 0x0000_f52e_ddc0_0000);
        assert_eq!(token.buffer_len(), 0x0010_0000);
        assert_eq!(token.as_str(), TOKEN);
    }

    #[test]
    fn token_rejects_malformed_values() {
        for bad in [
            "",
            "nothex:00100000:1:2:3:4:5",
            "0000f52eddc00000:nothex:1:2:3:4:5",
            "0000f52eddc00000:00000000:1:2:3:4:5",
            "0000f52eddc00000:00100000",
            "0000f52eddc00000:00100000:1:2:3",
            "0000f52eddc00000:00100000:1:2:3:4:\0",
        ] {
            assert!(
                matches!(RdmaToken::parse(bad), Err(Error::InvalidToken(_))),
                "{bad:?}"
            );
        }
    }
}
