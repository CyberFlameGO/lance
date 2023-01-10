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

use arrow_array::cast::downcast_array;
use arrow_array::{cast::as_primitive_array, RecordBatch};
use arrow_array::{FixedSizeListArray, Float32Array};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use futures::stream::StreamExt;
use futures::stream::{Stream};

use super::distance::euclidean_distance;
use super::SearchParams;
use crate::arrow::RecordBatchExt;
use crate::dataset::Dataset;
use crate::{Error, Result};

/// Flat Vector Index.
///
/// Flat index is a meta index, that does not build extra index structure,
/// and uses the full scan.
///
///
/// Reference:
///   - <https://github.com/facebookresearch/faiss/wiki/Faiss-indexes>
pub struct FlatIndex<'a> {
    dataset: &'a Dataset,

    /// Vector column to search for.
    column: String,
}

impl<'a> FlatIndex<'a> {
    /// Search the flat vector index.
    ///
    /// Returns a [RecordBatch] with Schema of:
    ///
    /// ```
    /// use arrow_schema::{Schema, Field, DataType};
    ///
    /// Schema::new(vec![
    ///   Field::new("_rowid", DataType::UInt64, false),
    ///   Field::new("score", DataType::Float32, false),
    /// ]);
    /// ```
    ///
    /// WARNINGS: only supports f32 now. add f64 later.
    pub async fn search(
        &self,
        params: &SearchParams,
    ) -> Result<Box<dyn Stream<Item = Result<RecordBatch>>>> {
        let stream = self
            .dataset
            .scan()
            .project(&[&self.column])?
            .with_row_id()
            .into_stream();

        let key_arr: &Float32Array = as_primitive_array(&params.key);
        let all_scores = stream
            .map(|b| async move {
                let schema = Arc::new(ArrowSchema::new(vec![
                    ArrowField::new("_rowid", DataType::UInt64, false),
                    ArrowField::new("score", DataType::Float32, false),
                ]));
                b.map(|batch| {
                    let value_arr = batch.column_with_name(&self.column).unwrap();
                    let targets = downcast_array::<FixedSizeListArray>(&value_arr);
                    let scores = euclidean_distance(key_arr, &targets)?;
                    RecordBatch::try_new(
                        schema,
                        vec![batch.column_with_name("_rowid").unwrap().clone(), scores],
                    ).map_err(|e| Error::from(e))
                })
            })
            .collect::<Vec<_>>()
            .await;
        todo!()
    }
}

#[cfg(test)]
mod tests {}
