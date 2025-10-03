# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.20)
project(video_streaming_demo_enhanced LANGUAGES NONE)

# Check if this application should be built
# Accept either the parent app name or individual server/client names
if(NOT (APP_video_streaming_demo_enhanced OR APP_video_streaming_demo_server OR APP_video_streaming_demo_client))
  return()
endif()

# Default to download datasets
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)

# Download the endoscopy sample data (required for replayer mode)
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(endoscopy
    URL nvidia/clara-holoscan/holoscan_endoscopy_sample_data:20230222
    DOWNLOAD_NAME holoscan_endoscopy_sample_data_20230222.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    GENERATE_GXF_ENTITIES
    GXF_ENTITIES_HEIGHT 480
    GXF_ENTITIES_WIDTH 854
    GXF_ENTITIES_CHANNELS 3
    GXF_ENTITIES_FRAMERATE 30
    ALL
  )
endif()

# Enable required operators manually
set(OP_video_streaming ON CACHE BOOL "Build the video_streaming operator" FORCE)
set(OP_streaming_client_enhanced ON CACHE BOOL "Build the streaming_client_enhanced operator" FORCE)
set(OP_streaming_server_enhanced ON CACHE BOOL "Build the streaming_server_enhanced operator" FORCE)

# Build client and server subdirectories
add_subdirectory(video_streaming_demo_client)
add_subdirectory(video_streaming_demo_server)

# Add integration testing
if(BUILD_TESTING)
  # Copy integration test script
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/integration_test.sh"
    "${CMAKE_BINARY_DIR}/integration_test.sh"
    COPYONLY
  )

  # Make script executable
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/integration_test.sh"
       DESTINATION "${CMAKE_BINARY_DIR}"
       FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

  # Add integration test for video streaming demo
  add_test(NAME video_streaming_integration_test
           COMMAND bash -c "
             echo '=== Custom Integration Test ==='
             echo 'Starting server and client in same container...'

             # Initialize variables
             SERVER_SUCCESS=0
             CLIENT_SUCCESS=0

             # Start server in background
             echo 'Starting streaming server...'
             ./applications/video_streaming_demo_enhanced/video_streaming_demo_server/cpp/streaming_server_demo_enhanced --config applications/video_streaming_demo_enhanced/video_streaming_demo_server/cpp/streaming_server_demo.yaml &
             SERVER_PID=\$!

             # Wait for server to start
             echo 'Waiting for server to initialize...'
             sleep 10

             # Check if server is running and has been running for a while
             echo 'Checking if server is running and stable...'
             for i in {1..15}; do
               # Check if server process is still running
               if kill -0 \$SERVER_PID 2>/dev/null; then
                 echo \"Server process is running (attempt \$i/15)\"
                 sleep 2
                 # After 10 seconds, consider server stable
                 if [ \$i -ge 5 ]; then
                   echo '✓ Server is running and stable'
                   SERVER_SUCCESS=1
                   break
                 fi
               else
                 echo \"Server process died (attempt \$i/15)\"
                 break
               fi
             done

             if [ \$SERVER_SUCCESS -eq 1 ]; then
               echo '✓ Server started successfully'

               # Start client in background and let it stream for a while
               echo 'Starting streaming client...'
               ./applications/video_streaming_demo_enhanced/video_streaming_demo_client/cpp/streaming_client_demo_enhanced --config applications/video_streaming_demo_enhanced/video_streaming_demo_client/cpp/streaming_client_demo_replayer.yaml &
               CLIENT_PID=\$!

               # Let client stream for 30 seconds
               echo 'Letting client stream for 30 seconds...'
               sleep 30

               # Check if client is still running (indicating successful connection)
               if kill -0 \$CLIENT_PID 2>/dev/null; then
                 echo '✓ Client is still running - connection successful'
                 CLIENT_SUCCESS=1

                 # Stop client gracefully
                 echo 'Stopping client...'
                 kill -TERM \$CLIENT_PID 2>/dev/null || true
                 sleep 2
                 kill -KILL \$CLIENT_PID 2>/dev/null || true
               else
                 echo '✗ Client stopped unexpectedly - connection failed'
                 CLIENT_SUCCESS=0
               fi
             else
               echo '✗ Server failed to start'
               CLIENT_SUCCESS=0
             fi

             # Clean up server
             echo 'Cleaning up server process...'
             kill -TERM \$SERVER_PID 2>/dev/null || true
             sleep 2
             kill -KILL \$SERVER_PID 2>/dev/null || true

             # Overall result
             if [ \$SERVER_SUCCESS -eq 1 ] && [ \$CLIENT_SUCCESS -eq 1 ]; then
               echo '✓ Integration test PASSED'
               exit 0
             else
               echo '✗ Integration test FAILED'
               exit 1
             fi
           "
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  set_tests_properties(video_streaming_integration_test PROPERTIES
                       TIMEOUT 300
                       PASS_REGULAR_EXPRESSION "Integration test PASSED"
                       FAIL_REGULAR_EXPRESSION "Integration test FAILED")

  # Python Integration Test
  if(HOLOHUB_BUILD_PYTHON)
    add_test(NAME video_streaming_integration_test_python
             COMMAND bash -c "
               set -e
               export EnableHybridMode=1
               export PYTHONPATH=${CMAKE_BINARY_DIR}/python/lib:${CMAKE_BINARY_DIR}/video_streaming_demo_server/python/lib:${CMAKE_BINARY_DIR}/video_streaming_demo_client/python/lib:\$PYTHONPATH

               echo '=== Starting Python Integration Test ==='
               echo 'Using PYTHONPATH: '\$PYTHONPATH

               SERVER_SUCCESS=0
               CLIENT_SUCCESS=0

               # Start Python server in background
               echo 'Starting Python streaming server...'
               cd ${CMAKE_BINARY_DIR}
               python3 streaming_server_demo.py --port 48010 --width 854 --height 480 > server_python.log 2>&1 &
               SERVER_PID=\$!

               # Wait for server to start
               echo 'Waiting for Python server to initialize...'
               sleep 10

               # Check if server is running and stable
               echo 'Checking if Python server is running and stable...'
               for i in {1..15}; do
                 if kill -0 \$SERVER_PID 2>/dev/null; then
                   echo \"Python server process is running (attempt \$i/15)\"
                   sleep 2
                   if [ \$i -ge 5 ]; then
                     echo '✓ Python server is running and stable'
                     SERVER_SUCCESS=1
                     break
                   fi
                 else
                   echo \"Python server process died (attempt \$i/15)\"
                   break
                 fi
               done

               if [ \$SERVER_SUCCESS -eq 1 ]; then
                 echo '✓ Python server started successfully'

                 # Start Python client in background
                 echo 'Starting Python streaming client...'
                 python3 streaming_client_demo.py --port 48010 --source replayer --width 854 --height 480 --no-viz > client_python.log 2>&1 &
                 CLIENT_PID=\$!

                 # Let client stream for 30 seconds
                 echo 'Letting Python client stream for 30 seconds...'
                 sleep 30

                 # Check if client is still running
                 if kill -0 \$CLIENT_PID 2>/dev/null; then
                   echo '✓ Python client is still running - connection successful'
                   CLIENT_SUCCESS=1

                   # Stop client gracefully
                   echo 'Stopping Python client...'
                   kill -TERM \$CLIENT_PID 2>/dev/null || true
                   sleep 2
                   kill -KILL \$CLIENT_PID 2>/dev/null || true
                 else
                   echo '✗ Python client stopped unexpectedly - connection failed'
                   CLIENT_SUCCESS=0
                 fi
               else
                 echo '✗ Python server failed to start'
                 CLIENT_SUCCESS=0
               fi

               # Clean up server
               echo 'Cleaning up Python server process...'
               kill -TERM \$SERVER_PID 2>/dev/null || true
               sleep 2
               kill -KILL \$SERVER_PID 2>/dev/null || true

               # Show logs on failure
               if [ \$SERVER_SUCCESS -eq 0 ] || [ \$CLIENT_SUCCESS -eq 0 ]; then
                 echo '=== Python Server Log ==='
                 cat server_python.log || true
                 echo '=== Python Client Log ==='
                 cat client_python.log || true
               fi

               # Overall result
               if [ \$SERVER_SUCCESS -eq 1 ] && [ \$CLIENT_SUCCESS -eq 1 ]; then
                 echo '✓ Python Integration test PASSED'
                 exit 0
               else
                 echo '✗ Python Integration test FAILED'
                 exit 1
               fi
             "
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    set_tests_properties(video_streaming_integration_test_python PROPERTIES
                         TIMEOUT 300
                         PASS_REGULAR_EXPRESSION "Python Integration test PASSED"
                         FAIL_REGULAR_EXPRESSION "Python Integration test FAILED")
  endif()
endif()