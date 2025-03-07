// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Optimized local I/Os

use std::fs::File;
use std::ops::Range;
use std::sync::Arc;
// TODO: worry about windows later.
use std::os::unix::fs::FileExt;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use object_store::path::Path;

use super::object_reader::ObjectReader;
use crate::Result;

/// [ObjectReader] for local file system.
pub struct LocalObjectReader {
    /// File handler.
    file: Arc<File>,

    /// Preferred I/O size, in bytes.
    ///
    /// It could be the block size for local SSD.
    prefetch_size: usize,
}

/// Default prefetch size for local SSD.
const PREFETCH_SIZE: usize = 4096;

impl LocalObjectReader {
    /// Open a local object reader, with default prefetch size.
    pub fn open(path: &Path) -> Result<Box<dyn ObjectReader>> {
        Self::open_with_prefetch(path, PREFETCH_SIZE)
    }

    /// Open a local object reader, with specified `prefetch` size.
    pub fn open_with_prefetch(path: &Path, prefetch: usize) -> Result<Box<dyn ObjectReader>> {
        let local_path = format!("/{path}");
        Ok(Box::new(Self {
            file: File::open(local_path)?.into(),
            prefetch_size: prefetch,
        }))
    }
}

#[async_trait]
impl ObjectReader for LocalObjectReader {
    fn prefetch_size(&self) -> usize {
        self.prefetch_size
    }

    /// Returns the file size.
    async fn size(&self) -> Result<usize> {
        Ok(self.file.metadata()?.len() as usize)
    }

    /// Reads a range of data.
    ///
    /// TODO: return [arrow_buffer::Buffer] to avoid one memory copy from Bytes to Buffer.
    async fn get_range(&self, range: Range<usize>) -> Result<Bytes> {
        let file = self.file.clone();
        tokio::task::spawn_blocking(move || {
            let mut buf = BytesMut::zeroed(range.len());
            file.read_at(buf.as_mut(), range.start as u64)?;
            Ok(buf.freeze())
        })
        .await?
    }
}
