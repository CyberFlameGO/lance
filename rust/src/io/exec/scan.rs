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

use std::cmp::min;
use std::ops::{Range, RangeFrom};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use futures::stream::Stream;
use object_store::path::Path;
use tokio::sync::mpsc::{self, Receiver};
use tokio::task::JoinHandle;

use super::{ExecNode, NodeType};
use crate::format::Manifest;
use crate::io::{FileReader, ObjectStore};
use crate::{datatypes::Schema, format::Fragment};
use crate::{Error, Result};

/// Dataset Scan Node.
pub(crate) struct Scan {
    rx: Receiver<Result<RecordBatch>>,

    _io_thread: JoinHandle<()>,
}

impl Scan {
    /// Create a new scan node.
    pub fn new(
        object_store: Arc<ObjectStore>,
        data_dir: Path,
        fragments: Arc<Vec<Fragment>>,
        projection: &Schema,
        manifest: Arc<Manifest>,
        prefetch_size: usize,
        with_row_id: bool,
        limit: Option<i64>,
        offset: Option<i64>,
    ) -> Self {
        let (tx, rx) = mpsc::channel(prefetch_size);

        let projection = projection.clone();
        let io_thread = tokio::spawn(async move {
            let mut offset: u32 = offset.unwrap_or(0) as u32;
            let mut nrows_togo: u32 = limit.map(|limit| limit as u32).unwrap_or(u32::MAX);
            for frag in fragments.as_ref() {
                if tx.is_closed() || nrows_togo <= 0 {
                    return;
                }
                let data_file = &frag.files[0];
                let path = data_dir.child(data_file.path.clone());
                let reader = match FileReader::try_new_with_fragment(
                    &object_store,
                    &path,
                    frag.id,
                    Some(manifest.as_ref()),
                )
                .await
                {
                    Ok(mut r) => {
                        r.set_projection(projection.clone());
                        r.with_row_id(with_row_id);
                        r
                    }
                    Err(e) => {
                        tx.send(Err(Error::IO(format!(
                            "Failed to open file: {}: {}",
                            path, e
                        ))))
                        .await
                        .expect("Scanner sending error message");
                        // Stop reading.
                        break;
                    }
                };

                let nrows_file = reader.len() as u32;
                if offset > nrows_file {
                    offset -= nrows_file;
                    continue;
                }
                let start = reader.index_to_batch(offset);
                let end = reader.index_to_batch(min(nrows_file - 1, nrows_togo));

                for batch_id in start.batch_id..=end.batch_id {
                    let batch_start = if batch_id == start.batch_id {
                        start.offsets[0]
                    } else {
                        0
                    };
                    let batch = if batch_id == end.batch_id {
                        let params: Range<usize> =
                            batch_start as usize..(end.offsets[0] + 1) as usize;
                        reader.read_batch(batch_id, params).await
                    } else {
                        let params: RangeFrom<usize> = batch_start as usize..;
                        reader.read_batch(batch_id, params).await
                    };
                    if let Ok(b) = &batch {
                        nrows_togo -= b.num_rows() as u32;
                    }
                    if tx.is_closed() {
                        break;
                    }
                    if tx.send(batch).await.is_err() {
                        break;
                    }
                }
                offset = 0;
            }
            drop(tx)
        });

        Self {
            rx,
            _io_thread: io_thread, // Drop the background I/O thread with the stream.
        }
    }
}

impl ExecNode for Scan {
    fn node_type(&self) -> NodeType {
        NodeType::Scan
    }
}

impl Stream for Scan {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::into_inner(self).rx.poll_recv(cx)
    }
}
