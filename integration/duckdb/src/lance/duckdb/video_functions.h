// Copyright 2022 Lance Authors
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
//

#pragma once

#include <duckdb.hpp>
#include <duckdb/parser/parsed_data/create_table_function_info.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <vector>

/**
 * Use duckdb to work with videos
 */
namespace lance::duckdb {

std::vector<std::unique_ptr<::duckdb::CreateTableFunctionInfo>> GetVideoTableFunctions();

/**
 * Replacement scan for mp4 files (e.g, "SELECT * FROM 'my_video.mp4'")
 * @param context
 * @param table_name  this is the video uri
 * @param data
 * @return
 */
std::unique_ptr<::duckdb::TableFunctionRef> VideoScanReplacement(
    ::duckdb::ClientContext &context,
    const ::std::string &table_name,
    ::duckdb::ReplacementScanData *data);
}
