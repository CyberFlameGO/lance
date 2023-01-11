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

//! Compute distance
//!
//! Support method:
//!  - Euclidean Distance (L2)

use std::sync::Arc;

use arrow_arith::aggregate::sum;
use arrow_arith::arithmetic::{multiply, subtract};
use arrow_array::{Array, FixedSizeListArray, Float32Array};
use arrow_schema::DataType;

use crate::Result;

/// Euclidean distance from a point to a list of points.
pub fn euclidean_distance(
    from: &Float32Array,
    to: &FixedSizeListArray,
) -> Result<Arc<Float32Array>> {
    assert_eq!(from.len(), to.value_length() as usize);
    assert_eq!(to.value_type(), DataType::Float32);

    // Naive implementation.
    // TODO: benchmark and use SIMD or BLAS
    let scores: Float32Array = (0..to.len())
        .map(|idx| to.value(idx))
        .map(|left| {
            let arr = left.as_any().downcast_ref::<Float32Array>().unwrap();
            let mut sub = subtract(arr, from).unwrap();
            sub = multiply(&sub, &sub).unwrap();
            sum(&sub).unwrap().sqrt()
        })
        .collect();
    Ok(Arc::new(scores))
}

#[cfg(test)]
mod tests {
    use arrow_array::types::Float32Type;

    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let mat = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vec![
                Some(vec![Some(1.0), Some(2.0), Some(3.0)]),
                Some(vec![Some(2.0), Some(3.0), Some(4.0)]),
                Some(vec![Some(3.0), Some(4.0), Some(5.0)]),
                Some(vec![Some(4.0), Some(5.0), Some(6.0)]),
            ],
            3,
        );
        let point = Float32Array::from(vec![2.0, 3.0, 4.0]);
        let scores = euclidean_distance(&point, &mat).unwrap();

        assert_eq!(
            scores.as_ref(),
            &Float32Array::from(vec![3.0_f32.sqrt(), 0_f32, 3.0_f32.sqrt(), 12.0_f32.sqrt()])
        );
    }
}
