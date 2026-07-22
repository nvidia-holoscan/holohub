/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

#include "mhd_loader.hpp"
#include "volume.hpp"

namespace holoscan::ops {

class MhdLoaderHeaderTest : public testing::Test {
 protected:
  void SetUp() override {
    header_path_ = std::filesystem::temp_directory_path() / "holohub_mhd_loader_test.mhd";
    raw_path_ = std::filesystem::temp_directory_path() / "holohub_mhd_loader_test.raw";
  }

  void TearDown() override {
    std::error_code error;
    std::filesystem::remove(header_path_, error);
    std::filesystem::remove(raw_path_, error);
  }

  bool load_header(const std::string& header) {
    std::ofstream header_file(header_path_);
    if (!header_file.is_open()) {
      ADD_FAILURE() << "Failed to open test MHD header " << header_path_;
      return false;
    }

    header_file << header;
    if (!header_file.good()) {
      ADD_FAILURE() << "Failed to write test MHD header " << header_path_;
      return false;
    }

    header_file.close();
    if (header_file.fail()) {
      ADD_FAILURE() << "Failed to close test MHD header " << header_path_;
      return false;
    }

    Volume volume;
    return load_mhd(header_path_.string(), volume);
  }

  std::filesystem::path header_path_;
  std::filesystem::path raw_path_;
};

TEST_F(MhdLoaderHeaderTest, RejectsTooFewDimensions) {
  EXPECT_FALSE(load_header("NDims = 3\nDimSize = 10 20\n"));
}

TEST_F(MhdLoaderHeaderTest, RejectsInvalidDimensionCount) {
  EXPECT_FALSE(load_header("NDims = 3abc\nDimSize = 10 20 30\n"));
}

TEST_F(MhdLoaderHeaderTest, RejectsTooManyDimensions) {
  EXPECT_FALSE(load_header("NDims = 3\nDimSize = 10 20 30 40\n"));
}

TEST_F(MhdLoaderHeaderTest, RejectsNonPositiveDimensions) {
  EXPECT_FALSE(load_header("NDims = 3\nDimSize = 10 0 30\n"));
  EXPECT_FALSE(load_header("NDims = 3\nDimSize = 10 -20 30\n"));
}

TEST_F(MhdLoaderHeaderTest, RejectsTooFewSpacingValues) {
  EXPECT_FALSE(load_header("NDims = 3\nDimSize = 10 20 30\nElementSpacing = 1 1\n"));
}

TEST_F(MhdLoaderHeaderTest, RejectsTooManySpacingValues) {
  EXPECT_FALSE(load_header("NDims = 3\nDimSize = 10 20 30\nElementSpacing = 1 1 1 1\n"));
}

TEST_F(MhdLoaderHeaderTest, RejectsNonPositiveSpacing) {
  EXPECT_FALSE(load_header("NDims = 3\nDimSize = 10 20 30\nElementSpacing = 1 0 1\n"));
  EXPECT_FALSE(load_header("NDims = 3\nDimSize = 10 20 30\nElementSpacing = 1 -1 1\n"));
}

TEST_F(MhdLoaderHeaderTest, RejectsMissingRequiredFields) {
  EXPECT_FALSE(load_header("NDims = 3\nDimSize = 10 20 30\n"));
}

TEST_F(MhdLoaderHeaderTest, RejectsOverflowingVolumeSize) {
  EXPECT_FALSE(load_header("NDims = 3\n"
                           "DimSize = 2147483647 2147483647 2147483647\n"
                           "ElementType = MET_FLOAT\n"
                           "ElementDataFile = unused.raw\n"));
}

TEST_F(MhdLoaderHeaderTest, RejectsTruncatedRawPayload) {
  std::ofstream raw_file(raw_path_, std::ios::binary);
  ASSERT_TRUE(raw_file.is_open());
  raw_file.write("1234567", 7);
  ASSERT_TRUE(raw_file.good());
  raw_file.close();
  ASSERT_FALSE(raw_file.fail());

  EXPECT_FALSE(load_header("NDims = 3\n"
                           "DimSize = 2 2 2\n"
                           "ElementType = MET_UCHAR\n"
                           "ElementDataFile = holohub_mhd_loader_test.raw\n"));
}

}  // namespace holoscan::ops
