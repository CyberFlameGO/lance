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

//! Approximated Nearest Neighbor index
//!

mod distance;
mod flat;
mod ivf_pq;
mod sort;

use arrow_array::ArrayRef;
pub use flat::FlatIndex;
pub use sort::find_min_k;

/// Search parameters for the ANN indices
pub struct SearchParams {
    /// The vector to be searched.
    pub key: ArrayRef,
    /// Top k results to return.
    pub k: usize,
    /// number of probs to load and search
    pub nprob: usize,
}
