# SPDX-FileCopyrightText:  Copyright (c) 2022, DELTACAST.TV. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Find the main root of the SDK
find_path(Prohawk_SDK_DIR NAMES PTGDE/CPTGDE.h PATHS /usr/local/phruntime REQUIRED)

find_path(Prohawk_INCLUDE_DIR NAMES CPTGDE.h PATHS ${Prohawk_SDK_DIR}/PTGDE REQUIRED)
mark_as_advanced(Prohawk_INCLUDE_DIR)

find_library(ptgde
    NAMES libptgde.so
    PATHS /usr/lib ${Prohawk_SDK_DIR}
    REQUIRED
)

if(ptgde)
  add_library(Prohawk::ptgde SHARED IMPORTED)
  set_target_properties(Prohawk::ptgde PROPERTIES
            IMPORTED_LOCATION "${ptgde}"
            INTERFACE_INCLUDE_DIRECTORIES "${Prohawk_INCLUDE_DIR};${Prohawk_SDK_DIR}"
        )
  set(Prohawk_FOUND 1)
endif()


# Generate Prohawk_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Prohawk
    FOUND_VAR Prohawk_FOUND
    REQUIRED_VARS ptgde
)

