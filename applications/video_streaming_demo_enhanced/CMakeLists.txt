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
if(NOT APP_video_streaming_demo_enhanced)
  return()
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
             
             # Start server in background
             echo 'Starting streaming server...'
             ./applications/video_streaming_demo_enhanced/video_streaming_demo_server/cpp/streaming_server_demo_enhanced --config applications/video_streaming_demo_enhanced/video_streaming_demo_server/cpp/streaming_server_demo.yaml &
             SERVER_PID=\$!
             
             # Wait for server to start
             echo 'Waiting for server to initialize...'
             sleep 10
             
             # Check if server is listening
             echo 'Checking if server is listening on port 48010...'
             for i in {1..30}; do
               if netstat -tlnp 2>/dev/null | grep -q ':48010'; then
                 echo '✓ Server is listening on port 48010'
                 SERVER_SUCCESS=1
                 break
               else
                 echo \"Waiting for server to start... (attempt \$i/30)\"
                 sleep 2
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
endif()