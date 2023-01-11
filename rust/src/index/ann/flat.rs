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
use arrow_array::{
    cast::{as_primitive_array, as_struct_array},
    RecordBatch,
};
use arrow_array::{FixedSizeListArray, Float32Array, StructArray};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use arrow_select::{concat::concat_batches, take::take};
use futures::stream::StreamExt;

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
/// Reference:
///   - <https://github.com/facebookresearch/faiss/wiki/Faiss-indexes>
pub struct FlatIndex<'a> {
    dataset: &'a Dataset,

    /// Vector column to search for.
    column: String,
}

impl<'a> FlatIndex<'a> {
    /// Create the flat index.
    pub fn new(dataset: &'a Dataset, column: String) -> Self {
        Self { dataset, column }
    }

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
    /// WARNINGS:
    ///  - Only supports f32 now. add f64 later.
    pub async fn search(&self, params: &SearchParams) -> Result<RecordBatch> {
        let stream = self
            .dataset
            .scan()
            .project(&[&self.column])?
            .with_row_id()
            .into_stream();

        let key_arr: &Float32Array = as_primitive_array(&params.key);
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("_rowid", DataType::UInt64, false),
            ArrowField::new("score", DataType::Float32, false),
        ]));
        let all_scores = stream
            .then(|b| async {
                if let Ok(batch) = b {
                    let value_arr = batch.column_with_name(&self.column).unwrap();
                    let targets = downcast_array::<FixedSizeListArray>(&value_arr);
                    let scores = euclidean_distance(key_arr, &targets)?;
                    // println!("Scores: {:?}", scores);
                    RecordBatch::try_new(
                        schema.clone(),
                        vec![batch.column_with_name("_rowid").unwrap().clone(), scores],
                    )
                    .map_err(|e| Error::from(e))
                } else {
                    b
                }
            })
            .collect::<Vec<_>>()
            .await;
        let scores = concat_batches(
            &schema,
            all_scores
                .iter()
                .map(|s| s.as_ref().unwrap().clone())
                .collect::<Vec<_>>()
                .as_slice(),
        )?;
        let scores_arr = scores.column_with_name("score").unwrap();
        let indices = sort_to_indices(scores_arr, None, Some(params.k))?;

        let struct_arr = StructArray::from(scores);
        let taken_scores = take(&struct_arr, &indices, None)?;
        // Ok(scores)
        Ok(as_struct_array(&taken_scores).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;
    use std::iter::repeat_with;

    use arrow_array::Float32Array;
    use rand::Rng;

    pub fn generate_random_array(n: usize) -> Arc<Float32Array> {
        let mut rng = rand::thread_rng();
        Arc::new(Float32Array::from(
            repeat_with(|| rng.gen::<f32>())
                .take(n)
                .collect::<Vec<f32>>(),
        ))
    }

    #[tokio::test]
    async fn test_flat_index() {
        let dataset = Dataset::open("/Users/lei/work/lance/rust/vec_data")
            .await
            .unwrap();
        println!("Dataset schema: {:?}", dataset.schema());

        let index = FlatIndex::new(&dataset, "vec".to_string());
        let params = SearchParams {
            key: generate_random_array(1024),
            k: 10,
            nprob: 0,
        };
        let scores = index.search(&params).await.unwrap();
        println!("scores: {:?}\n", scores);
    }
}
