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

use std::sync::Arc;

use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema, SchemaRef};
use futures::stream::Stream;
use object_store::path::Path;
use tokio::sync::mpsc::{self, Receiver};

use super::Dataset;
use crate::datatypes::{Field, Schema};
use crate::format::{Fragment, Manifest};
use crate::io::{FileReader, ObjectStore};
use crate::{Error, Result};

/// Dataset Scanner
///
/// ```rust,ignore
/// let dataset = Dataset::open(uri).await.unwrap();
/// let stream = dataset.scan()
///     .project(&["col", "col2.subfield"]).unwrap()
///     .limit(10)
///     .into_stream();
/// stream
///   .map(|batch| batch.num_rows())
///   .buffered(16)
///   .sum()
/// ```
pub struct Scanner<'a> {
    dataset: &'a Dataset,

    projections: Schema,

    // filter: how to present filter
    limit: Option<i64>,
    offset: Option<i64>,

    // If set true, returns the row ID from the dataset alongside with the
    // actual data.
    with_row_id: bool,

    fragments: Vec<Fragment>,
}

impl<'a> Scanner<'a> {
    pub fn new(dataset: &'a Dataset) -> Self {
        Self {
            dataset,
            projections: dataset.schema().clone(),
            limit: None,
            offset: None,
            with_row_id: false,
            fragments: dataset.fragments().to_vec(),
            with_row_id: false,
        }
    }

    /// Projection.
    ///
    /// Only seelect the specific columns. If not specifid, all columns will be scanned.
    pub fn project(&mut self, columns: &[&str]) -> Result<&mut Self> {
        self.projections = self.dataset.schema().project(columns)?;
        Ok(self)
    }

    /// Set limit and offset.
    pub fn limit(&mut self, limit: i64, offset: Option<i64>) -> &mut Self {
        self.limit = Some(limit);
        self.offset = offset;
        self
    }

    /// Instruct the scanner to return the `_rowid` meta column from the dataset.
    pub fn with_row_id(&mut self) -> &mut Self {
        self.with_row_id = true;
        self
    }

    /// The schema of the output, a.k.a, projection schema.
    pub fn schema(&self) -> SchemaRef {
        Arc::new(ArrowSchema::from(&self.projections))
    }

    /// Create a stream of this Scanner.
    ///
    /// TODO: implement as IntoStream/IntoIterator.
    pub fn into_stream(&self) -> ScannerStream {
        const PREFECTH_SIZE: usize = 32;
        let object_store = self.dataset.object_store.clone();

        let data_dir = self.dataset.data_dir().clone();
        let fragments = self.fragments.clone();
        let manifest = self.dataset.manifest.clone();

        ScannerStream::new(
            object_store,
            data_dir,
            fragments,
            manifest,
            PREFECTH_SIZE,
            &self.projections,
            self.with_row_id,
        )
    }
}

pub struct ScannerStream {
    rx: Receiver<Result<RecordBatch>>,
}

impl ScannerStream {
    fn new(
        object_store: Arc<ObjectStore>,
        data_dir: Path,
        fragments: Vec<Fragment>,
        manifest: Arc<Manifest>,
        prefetch_size: usize,
        schema: &Schema,
        with_row_id: bool,
    ) -> Self {
        let (tx, rx) = mpsc::channel(prefetch_size);

        let schema = schema.clone();
        let mut return_schema = schema.clone();
        if with_row_id {
            return_schema.fields.push(
                Field::try_from(&ArrowField::new("_rowid", DataType::UInt64, false)).unwrap(),
            );
        }
        tokio::spawn(async move {
            for frag in &fragments {
                let mut row_count = 0;
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
                        r.set_projection(schema.clone());
                        r.with_row_id(with_row_id);
                        r
                    }
                    Err(e) => {
                        tx.send(Err(Error::IO(format!(
                            "Failed to open file: {}: {}",
                            path.to_string(),
                            e.to_string()
                        ))))
                        .await
                        .unwrap();
                        // Stop reading.
                        break;
                    }
                };
                for batch_id in 0..reader.num_batches() {
                    println!("Read batch: {} channel cap={}", batch_id, tx.capacity());
                    let batch = reader.read_batch(batch_id as i32).await.map(|b| {
                        if with_row_id {
                            // Add a meta column;
                            let mut columns = b.columns().to_vec();
                            columns.push(Arc::new(UInt64Array::from(
                                (row_count..row_count + b.num_rows() as u64).collect::<Vec<u64>>(),
                            )));
                            RecordBatch::try_new(
                                Arc::new(ArrowSchema::try_from(&return_schema).unwrap()),
                                columns.to_vec(),
                            )
                            .unwrap()
                        } else {
                            b
                        }
                    });
                    batch.as_ref().and_then(|b| {
                        row_count += b.num_rows() as u64;
                        Ok(b)
                    });
                    tx.send(batch).await.unwrap();
                }
            }
            drop(tx)
        });
        Self { rx }
    }
}

impl Stream for ScannerStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::into_inner(self).rx.poll_recv(cx)
    }
}
