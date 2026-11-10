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

//! Just enough HTTP/1.1 for the examples: one request or response head, no
//! bodies (the bytes move over RDMA).
//!
//! A chunk travels as `GET /<key>` or `PUT /<key>` with
//! `x-amz-rdma-token: <token>` and `range: bytes=<first>-<last>` (the chunk's
//! place in the object). The response is `200 OK` with
//! `x-amz-rdma-reply: 200` and `x-amz-rdma-bytes-transferred: <n>`.

#![allow(dead_code, reason = "each example uses part of it")]

use std::io::{self, BufRead, Write};

/// A request or status line and its headers.
pub struct Head {
    pub line: String,
    pub headers: Vec<(String, String)>,
}

impl Head {
    /// The first header named `name` (case-insensitive).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// Writes the head and the blank line that ends it.
    pub fn write_to(&self, out: &mut impl Write) -> io::Result<()> {
        let mut text = format!("{}\r\n", self.line);
        for (name, value) in &self.headers {
            text.push_str(&format!("{name}: {value}\r\n"));
        }
        text.push_str("content-length: 0\r\n\r\n");
        out.write_all(text.as_bytes())?;
        out.flush()
    }
}

/// Reads one head; `None` when the peer closed the connection first.
pub fn read_head(input: &mut impl BufRead) -> io::Result<Option<Head>> {
    let mut line = String::new();
    if input.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut head = Head {
        line: line.trim_end().to_owned(),
        headers: Vec::new(),
    };
    loop {
        line.clear();
        if input.read_line(&mut line)? == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated head",
            ));
        }
        let line = line.trim_end();
        if line.is_empty() {
            return Ok(Some(head));
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "malformed header"))?;
        head.headers
            .push((name.trim().to_owned(), value.trim().to_owned()));
    }
}

/// Parses `bytes=<first>-<last>` into `(offset, len)`.
pub fn parse_range(value: &str) -> Option<(u64, u64)> {
    let (first, last) = value.strip_prefix("bytes=")?.split_once('-')?;
    let (first, last): (u64, u64) = (first.parse().ok()?, last.parse().ok()?);
    Some((first, last.checked_sub(first)?.checked_add(1)?))
}

/// Formats `(offset, len)` as `bytes=<first>-<last>`.
pub fn range(offset: u64, len: u64) -> String {
    format!("bytes={offset}-{}", offset + len - 1)
}
