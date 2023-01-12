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

//! Secondary Index
//!

use async_trait::async_trait;

pub mod ann;
use crate::Result;

pub enum IndexType {
    // Preserve 0-100 for simple indices.

    // 100+ and up for vector index.
    /// Flat vector index.
    VectorFlat = 100,

    /// IVF_PQ vector index.
    VectorIvfPQ = 101,
}

/// Present an index.
#[async_trait]
pub trait Index {
    fn index_type() -> IndexType;

    async fn build(&self) -> Result<()>;
}
