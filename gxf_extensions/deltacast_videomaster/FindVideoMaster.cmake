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
find_path(VideoMaster_SDK_DIR NAMES VideoMasterHD_Core.h PATHS /usr/include/videomaster REQUIRED)

find_path(VideoMaster_INCLUDE_DIR NAMES VideoMasterHD_Core.h PATHS ${VideoMaster_SDK_DIR} REQUIRED)
mark_as_advanced(VideoMaster_INCLUDE_DIR)

find_library(videomasterhd_core
    NAMES libvideomasterhd.so
    PATHS /usr/lib
    REQUIRED
)

if(videomasterhd_core)
  add_library(VideoMaster::videomasterhd_core SHARED IMPORTED)
  set_target_properties(VideoMaster::videomasterhd_core PROPERTIES
            IMPORTED_LOCATION "${videomasterhd_core}"
            INTERFACE_INCLUDE_DIRECTORIES "${VideoMaster_INCLUDE_DIR}"
        )
  set(VideoMaster_FOUND 1)
endif()


# Generate VideoMaster_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VideoMaster
    FOUND_VAR VideoMaster_FOUND
    REQUIRED_VARS videomasterhd_core
)