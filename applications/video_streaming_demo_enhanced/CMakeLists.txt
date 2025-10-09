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
             echo '=== Enhanced Integration Test with Log Verification ==='
             echo 'Starting server and client with log capture...'

             # Initialize variables
             SERVER_SUCCESS=0
             CLIENT_SUCCESS=0
             STREAMING_SUCCESS=0

             # Create log files
             SERVER_LOG=\$(mktemp /tmp/server_log.XXXXXX)
             CLIENT_LOG=\$(mktemp /tmp/client_log.XXXXXX)

             echo \"Server log: \$SERVER_LOG\"
             echo \"Client log: \$CLIENT_LOG\"

             # Start server in background with log capture
             echo 'Starting streaming server...'
             ./applications/video_streaming_demo_enhanced/video_streaming_demo_server/cpp/streaming_server_demo_enhanced --config applications/video_streaming_demo_enhanced/video_streaming_demo_server/cpp/streaming_server_demo.yaml > \$SERVER_LOG 2>&1 &
             SERVER_PID=\$!

             # Wait for server to start and check logs
             echo 'Waiting for server to initialize...'
             sleep 10

             # Verify server started correctly
             if ! kill -0 \$SERVER_PID 2>/dev/null; then
               echo '✗ Server process died during startup'
               cat \$SERVER_LOG
               rm -f \$SERVER_LOG \$CLIENT_LOG
               exit 1
             fi

             echo '✓ Server process is running'

             # Start client in background with log capture
             echo 'Starting streaming client...'
             ./applications/video_streaming_demo_enhanced/video_streaming_demo_client/cpp/streaming_client_demo_enhanced --config applications/video_streaming_demo_enhanced/video_streaming_demo_client/cpp/streaming_client_demo_replayer.yaml > \$CLIENT_LOG 2>&1 &
             CLIENT_PID=\$!

             # Let streaming run for 30 seconds
             echo 'Letting streaming run for 30 seconds...'
             sleep 30

             # Stop client and server gracefully
             echo 'Stopping client...'
             kill -TERM \$CLIENT_PID 2>/dev/null || true
             sleep 2
             kill -KILL \$CLIENT_PID 2>/dev/null || true

             echo 'Stopping server...'
             kill -TERM \$SERVER_PID 2>/dev/null || true
             sleep 2
             kill -KILL \$SERVER_PID 2>/dev/null || true

             # Now verify logs for actual streaming behavior
             echo ''
             echo '=== Verifying Server Logs ==='

             # Check for client connection
             if grep -q 'Client connected' \$SERVER_LOG; then
               echo '✓ Server: Client connected'
               SERVER_SUCCESS=\$((SERVER_SUCCESS + 1))
             else
               echo '✗ Server: No client connection detected'
             fi

             # Check for upstream connection
             if grep -q 'Upstream connection established' \$SERVER_LOG; then
               echo '✓ Server: Upstream connection established'
               SERVER_SUCCESS=\$((SERVER_SUCCESS + 1))
             else
               echo '✗ Server: Upstream connection not established'
             fi

             # Check for downstream connection
             if grep -q 'Downstream connection established' \$SERVER_LOG; then
               echo '✓ Server: Downstream connection established'
               SERVER_SUCCESS=\$((SERVER_SUCCESS + 1))
             else
               echo '✗ Server: Downstream connection not established'
             fi

            # Check for upstream frame processing (client -> server)
            if grep -q 'Processing UNIQUE frame' \$SERVER_LOG; then
              FRAME_COUNT=\$(grep -c 'Processing UNIQUE frame' \$SERVER_LOG)
              if [ \$FRAME_COUNT -ge 100 ]; then
                echo \"✓ Server: StreamingServerUpstreamOp processed \$FRAME_COUNT unique frames\"
                SERVER_SUCCESS=\$((SERVER_SUCCESS + 1))
              else
                echo \"✗ Server: Only \$FRAME_COUNT frames processed (minimum: 100)\"
              fi
            else
              echo '✗ Server: StreamingServerUpstreamOp - No frames processed'
            fi

            # Check for downstream frame transmission (server -> client)
            if grep -q 'DOWNSTREAM: Processing tensor' \$SERVER_LOG; then
              DOWNSTREAM_COUNT=\$(grep -c 'DOWNSTREAM: Processing tensor' \$SERVER_LOG)
              if [ \$DOWNSTREAM_COUNT -ge 100 ]; then
                echo \"✓ Server: StreamingServerDownstreamOp processed \$DOWNSTREAM_COUNT tensors\"
                SERVER_SUCCESS=\$((SERVER_SUCCESS + 1))
              else
                echo \"✗ Server: Only \$DOWNSTREAM_COUNT tensors processed (minimum: 100)\"
              fi
            else
              echo '✗ Server: StreamingServerDownstreamOp - No tensors processed'
            fi

            # Check for frame processing statistics
            if grep -q 'Frame Processing Stats' \$SERVER_LOG; then
              echo '✓ Server: Frame processing statistics logged'
              SERVER_SUCCESS=\$((SERVER_SUCCESS + 1))
            else
              echo '✗ Server: No frame processing statistics logged'
            fi

             echo ''
             echo '=== Verifying Client Logs ==='

            # Check for successful frame sending
            if grep -q 'Frame sent successfully' \$CLIENT_LOG; then
              SENT_COUNT=\$(grep -c 'Frame sent successfully' \$CLIENT_LOG)
              if [ \$SENT_COUNT -ge 100 ]; then
                echo \"✓ Client: Sent \$SENT_COUNT frames successfully\"
                CLIENT_SUCCESS=\$((CLIENT_SUCCESS + 1))
              else
                echo \"✗ Client: Only \$SENT_COUNT frames sent (minimum: 100)\"
              fi
            else
              echo '✗ Client: No frames sent successfully'
            fi

            # Check for client frame reception (bidirectional verification)
            if grep -q 'CLIENT: Received frame' \$CLIENT_LOG; then
              RECEIVED_COUNT=\$(grep -c 'CLIENT: Received frame' \$CLIENT_LOG)
              if [ \$RECEIVED_COUNT -ge 100 ]; then
                echo \"✓ Client: Received \$RECEIVED_COUNT frames from server\"
                CLIENT_SUCCESS=\$((CLIENT_SUCCESS + 1))
              else
                echo \"✗ Client: Only \$RECEIVED_COUNT frames received (minimum: 100)\"
              fi
            else
              echo '✗ Client: No frames received from server'
            fi

            # Check for frame validation
             if grep -q 'Frame validation passed' \$CLIENT_LOG; then
               echo '✓ Client: Frame validation passed'
               CLIENT_SUCCESS=\$((CLIENT_SUCCESS + 1))
             else
               echo '✗ Client: Frame validation issues'
             fi

             # Check for client streaming start
             if grep -q 'STARTING STREAMING CLIENT' \$CLIENT_LOG; then
               echo '✓ Client: Streaming client started'
               CLIENT_SUCCESS=\$((CLIENT_SUCCESS + 1))
             else
               echo '✗ Client: Streaming client failed to start'
             fi

            echo ''
            echo '=== Test Results Summary ==='
            echo \"Server checks passed: \$SERVER_SUCCESS/6 (required: 6)\"
            echo \"Client checks passed: \$CLIENT_SUCCESS/4 (required: 4)\"

            # Determine overall success
            # Require ALL checks to pass for proper integration test
            if [ \$SERVER_SUCCESS -eq 6 ] && [ \$CLIENT_SUCCESS -eq 4 ]; then
               STREAMING_SUCCESS=1
               echo '✓ STREAMING VERIFICATION PASSED - Frames actually transmitted!'
             else
               echo '✗ STREAMING VERIFICATION FAILED - No actual frame transmission detected'
               echo ''
               echo '=== Server Log Excerpt ==='
               head -50 \$SERVER_LOG
               echo '...'
               tail -50 \$SERVER_LOG
               echo ''
               echo '=== Client Log Excerpt ==='
               head -50 \$CLIENT_LOG
               echo '...'
               tail -50 \$CLIENT_LOG
             fi

             # Clean up logs
             rm -f \$SERVER_LOG \$CLIENT_LOG

             # Overall result
             if [ \$STREAMING_SUCCESS -eq 1 ]; then
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