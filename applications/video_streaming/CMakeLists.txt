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
project(video_streaming LANGUAGES CXX)

# If APP_video_streaming is set, enable both client and server
if(APP_video_streaming)
  set(APP_video_streaming_client ON CACHE BOOL "Build the video_streaming_client application" FORCE)
  set(APP_video_streaming_server ON CACHE BOOL "Build the video_streaming_server application" FORCE)
endif()

# Build client and server applications
add_holohub_application(video_streaming_client DEPENDS
                        OPERATORS video_streaming_client)

add_holohub_application(video_streaming_server DEPENDS
                        OPERATORS video_streaming_server)

# Add integration testing
if(BUILD_TESTING)
  # Add C++ integration test
  if(HOLOHUB_BUILD_CPP)
    add_test(NAME video_streaming_integration_test_cpp
            COMMAND bash -c "
              echo '=== Integration Test with Log Verification ==='
              echo 'Starting server and client with log capture...'

              # Determine data directory path - use HOLOHUB_DATA_DIR if set, otherwise fall back to standard path
              DATA_DIR=\${HOLOHUB_DATA_DIR:-${HOLOHUB_DATA_DIR}}
              if [ -z \"\$DATA_DIR\" ]; then
                DATA_DIR=\"${CMAKE_SOURCE_DIR}/data\"
              fi

              # Find the actual endoscopy data directory
              if [ -d \"\$DATA_DIR/endoscopy\" ]; then
                DATA_PATH=\"\$DATA_DIR/endoscopy\"
              elif [ -d \"\$DATA_DIR\" ]; then
                DATA_PATH=\"\$DATA_DIR\"
              else
                echo \"✗ Error: Could not find data directory. Tried:\"
                echo \"  - \$DATA_DIR/endoscopy\"
                echo \"  - \$DATA_DIR\"
                exit 1
              fi

              echo \"Using data directory: \$DATA_PATH\"

              # Initialize variables
              TEST_FAILED=0
              SERVER_CHECKS_PASSED=0
              CLIENT_CHECKS_PASSED=0

              # Create log files
              SERVER_LOG=\$(mktemp /tmp/server_log.XXXXXX)
              CLIENT_LOG=\$(mktemp /tmp/client_log.XXXXXX)

              echo \"Server log: \$SERVER_LOG\"
              echo \"Client log: \$CLIENT_LOG\"

              # Start server in background with log capture
              echo 'Starting streaming server...'
              ./applications/video_streaming/video_streaming_server/cpp/streaming_server_demo --config applications/video_streaming/video_streaming_server/cpp/streaming_server_demo.yaml > \$SERVER_LOG 2>&1 &
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
               ./applications/video_streaming/video_streaming_client/cpp/video_streaming_client_demo_cpp --config applications/video_streaming/video_streaming_client/cpp/video_streaming_client_demo_replayer.yaml --data \"\$DATA_PATH\" > \$CLIENT_LOG 2>&1 &
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
                SERVER_CHECKS_PASSED=\$((SERVER_CHECKS_PASSED + 1))
              else
                echo '✗ Server: No client connection detected'
                TEST_FAILED=1
              fi

              # Check for upstream connection
              if grep -q 'Upstream connection established' \$SERVER_LOG; then
                echo '✓ Server: Upstream connection established'
                SERVER_CHECKS_PASSED=\$((SERVER_CHECKS_PASSED + 1))
              else
                echo '✗ Server: Upstream connection not established'
                TEST_FAILED=1
              fi

              # Check for downstream connection
              if grep -q 'Downstream connection established' \$SERVER_LOG; then
                echo '✓ Server: Downstream connection established'
                SERVER_CHECKS_PASSED=\$((SERVER_CHECKS_PASSED + 1))
              else
                echo '✗ Server: Downstream connection not established'
                TEST_FAILED=1
              fi

              # Check for upstream frame processing (client -> server)
              if grep -q 'Processing UNIQUE frame' \$SERVER_LOG; then
                FRAME_COUNT=\$(grep -c 'Processing UNIQUE frame' \$SERVER_LOG)
                if [ \$FRAME_COUNT -ge 100 ]; then
                  echo \"✓ Server: StreamingServerUpstreamOp processed \$FRAME_COUNT unique frames\"
                  SERVER_CHECKS_PASSED=\$((SERVER_CHECKS_PASSED + 1))
                else
                  echo \"✗ Server: Only \$FRAME_COUNT frames processed (minimum: 100)\"
                  TEST_FAILED=1
                fi
              else
                echo '✗ Server: StreamingServerUpstreamOp - No frames processed'
                TEST_FAILED=1
              fi

              # Check for downstream frame transmission (server -> client)
              if grep -q 'DOWNSTREAM: Processing tensor' \$SERVER_LOG; then
                DOWNSTREAM_COUNT=\$(grep -c 'DOWNSTREAM: Processing tensor' \$SERVER_LOG)
                if [ \$DOWNSTREAM_COUNT -ge 100 ]; then
                  echo \"✓ Server: StreamingServerDownstreamOp processed \$DOWNSTREAM_COUNT tensors\"
                  SERVER_CHECKS_PASSED=\$((SERVER_CHECKS_PASSED + 1))
                else
                  echo \"✗ Server: Only \$DOWNSTREAM_COUNT tensors processed (minimum: 100)\"
                  TEST_FAILED=1
                fi
              else
                echo '✗ Server: StreamingServerDownstreamOp - No tensors processed'
                TEST_FAILED=1
              fi

              # Check for frame processing statistics
              if grep -q 'Frame Processing Stats' \$SERVER_LOG; then
                echo '✓ Server: Frame processing statistics logged'
                SERVER_CHECKS_PASSED=\$((SERVER_CHECKS_PASSED + 1))
              else
                echo '✗ Server: No frame processing statistics logged'
                TEST_FAILED=1
              fi

              echo ''
              echo '=== Verifying Client Logs ==='

              # Check for successful frame sending
              if grep -q 'Frame sent successfully' \$CLIENT_LOG; then
                SENT_COUNT=\$(grep -c 'Frame sent successfully' \$CLIENT_LOG)
                if [ \$SENT_COUNT -ge 100 ]; then
                  echo \"✓ Client: Sent \$SENT_COUNT frames successfully\"
                  CLIENT_CHECKS_PASSED=\$((CLIENT_CHECKS_PASSED + 1))
                else
                  echo \"✗ Client: Only \$SENT_COUNT frames sent (minimum: 100)\"
                  TEST_FAILED=1
                fi
              else
                echo '✗ Client: No frames sent successfully'
                TEST_FAILED=1
              fi

              # Check for client frame reception (bidirectional verification)
              if grep -q 'CLIENT: Received frame' \$CLIENT_LOG; then
                RECEIVED_COUNT=\$(grep -c 'CLIENT: Received frame' \$CLIENT_LOG)
                if [ \$RECEIVED_COUNT -ge 100 ]; then
                  echo \"✓ Client: Received \$RECEIVED_COUNT frames from server\"
                  CLIENT_CHECKS_PASSED=\$((CLIENT_CHECKS_PASSED + 1))
                else
                  echo \"✗ Client: Only \$RECEIVED_COUNT frames received (minimum: 100)\"
                  TEST_FAILED=1
                fi
              else
                echo '✗ Client: No frames received from server'
                TEST_FAILED=1
              fi

              # Check for frame validation
              if grep -q 'Frame validation passed' \$CLIENT_LOG; then
                echo '✓ Client: Frame validation passed'
                CLIENT_CHECKS_PASSED=\$((CLIENT_CHECKS_PASSED + 1))
              else
                echo '✗ Client: Frame validation issues'
                TEST_FAILED=1
              fi

              # Check for client streaming start
              if grep -q 'STARTING STREAMING CLIENT' \$CLIENT_LOG; then
                echo '✓ Client: Streaming client started'
                CLIENT_CHECKS_PASSED=\$((CLIENT_CHECKS_PASSED + 1))
              else
                echo '✗ Client: Streaming client failed to start'
                TEST_FAILED=1
              fi

              echo ''
              echo '=== Test Results Summary ==='
              echo \"Server checks passed: \$SERVER_CHECKS_PASSED\"
              echo \"Client checks passed: \$CLIENT_CHECKS_PASSED\"

              # Determine overall success
              # Use failure flag: ANY check failure causes overall test failure
              if [ \$TEST_FAILED -eq 0 ]; then
                echo '✓ STREAMING VERIFICATION PASSED - All checks passed, frames transmitted!'
              else
                echo '✗ STREAMING VERIFICATION FAILED - One or more checks failed'
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

              # Overall result based on failure flag
              if [ \$TEST_FAILED -eq 0 ]; then
                echo '✓ Integration test PASSED'
                exit 0
              else
                echo '✗ Integration test FAILED'
                exit 1
              fi
            "
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    set_tests_properties(video_streaming_integration_test_cpp PROPERTIES
                        TIMEOUT 300
                        PASS_REGULAR_EXPRESSION "Integration test PASSED"
                        FAIL_REGULAR_EXPRESSION "Integration test FAILED")
  endif()

  # Add Python integration test
  if(HOLOHUB_BUILD_PYTHON)
    # Add integration test for Python video streaming demo
    add_test(NAME video_streaming_integration_test_python
             COMMAND bash -c "
               echo '=== Python Integration Test with Log Verification ==='
               echo 'Starting Python server and client with log capture...'

               # Set up PYTHONPATH to include built Python modules
               export PYTHONPATH=${CMAKE_BINARY_DIR}/python/lib:\$PYTHONPATH
               echo \"PYTHONPATH: \$PYTHONPATH\"

               # Determine data directory path - use HOLOHUB_DATA_DIR if set, otherwise fall back to standard path
               DATA_DIR=\${HOLOHUB_DATA_DIR:-${HOLOHUB_DATA_DIR}}
               if [ -z \"\$DATA_DIR\" ]; then
                 DATA_DIR=\"${CMAKE_SOURCE_DIR}/data\"
               fi

               # Find the actual endoscopy data directory
               if [ -d \"\$DATA_DIR/endoscopy\" ]; then
                 DATA_PATH=\"\$DATA_DIR/endoscopy\"
               elif [ -d \"\$DATA_DIR\" ]; then
                 DATA_PATH=\"\$DATA_DIR\"
               else
                 echo \"✗ Error: Could not find data directory. Tried:\"
                 echo \"  - \$DATA_DIR/endoscopy\"
                 echo \"  - \$DATA_DIR\"
                 exit 1
               fi

               echo \"Using data directory: \$DATA_PATH\"

               # Set environment variables for Python applications
               export HOLOSCAN_INPUT_PATH=\"\$DATA_PATH\"
               export HOLOHUB_DATA_PATH=\"\$DATA_DIR\"

               # Initialize variables
               TEST_FAILED=0
               SERVER_CHECKS_PASSED=0
               CLIENT_CHECKS_PASSED=0

               # Create log files
               SERVER_LOG=\$(mktemp /tmp/server_python_log.XXXXXX)
               CLIENT_LOG=\$(mktemp /tmp/client_python_log.XXXXXX)

               echo \"Python Server log: \$SERVER_LOG\"
               echo \"Python Client log: \$CLIENT_LOG\"

               # Start Python server in background with log capture
               echo 'Starting Python streaming server...'
               python3 streaming_server_demo.py --width 854 --height 480 > \$SERVER_LOG 2>&1 &
               SERVER_PID=\$!

               # Wait for server to start and check logs
               echo 'Waiting for Python server to initialize...'
               sleep 10

               # Verify server started correctly
               if ! kill -0 \$SERVER_PID 2>/dev/null; then
                 echo '✗ Python Server process died during startup'
                 cat \$SERVER_LOG
                 rm -f \$SERVER_LOG \$CLIENT_LOG
                 exit 1
               fi

               echo '✓ Python Server process is running'

               # Start Python client in background with log capture (using replayer for integration test)
               echo 'Starting Python streaming client...'
               python3 video_streaming_client_demo.py --source replayer --width 854 --height 480 > \$CLIENT_LOG 2>&1 &
               CLIENT_PID=\$!

               # Let streaming run for 30 seconds
               echo 'Letting Python streaming run for 30 seconds...'
               sleep 30

              # Stop client and server gracefully
              echo 'Stopping Python client...'
              kill -TERM \$CLIENT_PID 2>/dev/null || true
              sleep 2
              kill -KILL \$CLIENT_PID 2>/dev/null || true
              wait \$CLIENT_PID 2>/dev/null || true

              echo 'Stopping Python server...'
              kill -TERM \$SERVER_PID 2>/dev/null || true
              sleep 2
              kill -KILL \$SERVER_PID 2>/dev/null || true
              wait \$SERVER_PID 2>/dev/null || true

               # Now verify logs for actual streaming behavior
               echo ''
               echo '=== Verifying Python Server Logs ==='

               # Check for client connection
               if grep -q 'Client connected' \$SERVER_LOG; then
                 echo '✓ Python Server: Client connected'
                 SERVER_CHECKS_PASSED=\$((SERVER_CHECKS_PASSED + 1))
               else
                 echo '✗ Python Server: No client connection detected'
                 TEST_FAILED=1
               fi

               # Check for upstream connection
               if grep -q 'Upstream connection established' \$SERVER_LOG; then
                 echo '✓ Python Server: Upstream connection established'
                 SERVER_CHECKS_PASSED=\$((SERVER_CHECKS_PASSED + 1))
               else
                 echo '✗ Python Server: Upstream connection not established'
                 TEST_FAILED=1
               fi

               # Check for downstream connection
               if grep -q 'Downstream connection established' \$SERVER_LOG; then
                 echo '✓ Python Server: Downstream connection established'
                 SERVER_CHECKS_PASSED=\$((SERVER_CHECKS_PASSED + 1))
               else
                 echo '✗ Python Server: Downstream connection not established'
                 TEST_FAILED=1
               fi

              # Check for upstream frame processing (client -> server)
              if grep -q 'Processing UNIQUE frame' \$SERVER_LOG; then
                FRAME_COUNT=\$(grep -c 'Processing UNIQUE frame' \$SERVER_LOG)
                if [ \$FRAME_COUNT -ge 100 ]; then
                  echo \"✓ Python Server: StreamingServerUpstreamOp processed \$FRAME_COUNT unique frames\"
                  SERVER_CHECKS_PASSED=\$((SERVER_CHECKS_PASSED + 1))
                else
                  echo \"✗ Python Server: Only \$FRAME_COUNT frames processed (minimum: 100)\"
                  TEST_FAILED=1
                fi
              else
                echo '✗ Python Server: StreamingServerUpstreamOp - No frames processed'
                TEST_FAILED=1
              fi

              # Check for downstream frame transmission (server -> client)
              if grep -q 'DOWNSTREAM: Processing tensor' \$SERVER_LOG; then
                DOWNSTREAM_COUNT=\$(grep -c 'DOWNSTREAM: Processing tensor' \$SERVER_LOG)
                if [ \$DOWNSTREAM_COUNT -ge 100 ]; then
                  echo \"✓ Python Server: StreamingServerDownstreamOp processed \$DOWNSTREAM_COUNT tensors\"
                  SERVER_CHECKS_PASSED=\$((SERVER_CHECKS_PASSED + 1))
                else
                  echo \"✗ Python Server: Only \$DOWNSTREAM_COUNT tensors processed (minimum: 100)\"
                  TEST_FAILED=1
                fi
              else
                echo '✗ Python Server: StreamingServerDownstreamOp - No tensors processed'
                TEST_FAILED=1
              fi

              # Check for frame processing statistics
              if grep -q 'Frame Processing Stats' \$SERVER_LOG; then
                echo '✓ Python Server: Frame processing statistics logged'
                SERVER_CHECKS_PASSED=\$((SERVER_CHECKS_PASSED + 1))
              else
                echo '✗ Python Server: No frame processing statistics logged'
                TEST_FAILED=1
              fi

               echo ''
               echo '=== Verifying Python Client Logs ==='

              # Check for successful frame sending
              if grep -q 'Frame sent successfully' \$CLIENT_LOG; then
                SENT_COUNT=\$(grep -c 'Frame sent successfully' \$CLIENT_LOG)
                if [ \$SENT_COUNT -ge 100 ]; then
                  echo \"✓ Python Client: Sent \$SENT_COUNT frames successfully\"
                  CLIENT_CHECKS_PASSED=\$((CLIENT_CHECKS_PASSED + 1))
                else
                  echo \"✗ Python Client: Only \$SENT_COUNT frames sent (minimum: 100)\"
                  TEST_FAILED=1
                fi
              else
                echo '✗ Python Client: No frames sent successfully'
                TEST_FAILED=1
              fi

              # Check for client frame reception (bidirectional verification)
              if grep -q 'CLIENT: Received frame' \$CLIENT_LOG; then
                RECEIVED_COUNT=\$(grep -c 'CLIENT: Received frame' \$CLIENT_LOG)
                if [ \$RECEIVED_COUNT -ge 100 ]; then
                  echo \"✓ Python Client: Received \$RECEIVED_COUNT frames from server\"
                  CLIENT_CHECKS_PASSED=\$((CLIENT_CHECKS_PASSED + 1))
                else
                  echo \"✗ Python Client: Only \$RECEIVED_COUNT frames received (minimum: 100)\"
                  TEST_FAILED=1
                fi
              else
                echo '✗ Python Client: No frames received from server'
                TEST_FAILED=1
              fi

              # Check for frame validation
               if grep -q 'Frame validation passed' \$CLIENT_LOG; then
                 echo '✓ Python Client: Frame validation passed'
                 CLIENT_CHECKS_PASSED=\$((CLIENT_CHECKS_PASSED + 1))
               else
                 echo '✗ Python Client: Frame validation issues'
                 TEST_FAILED=1
               fi

               # Check for client streaming start
               if grep -qE 'STARTING STREAMING CLIENT|Starting Streaming Client Demo' \$CLIENT_LOG; then
                 echo '✓ Python Client: Streaming client started'
                 CLIENT_CHECKS_PASSED=\$((CLIENT_CHECKS_PASSED + 1))
               else
                 echo '✗ Python Client: Streaming client failed to start'
                 TEST_FAILED=1
               fi

              echo ''
              echo '=== Python Test Results Summary ==='
              echo \"Python Server checks passed: \$SERVER_CHECKS_PASSED\"
              echo \"Python Client checks passed: \$CLIENT_CHECKS_PASSED\"

              # Determine overall success
              # Use failure flag: ANY check failure causes overall test failure
              if [ \$TEST_FAILED -eq 0 ]; then
                 echo '✓ PYTHON STREAMING VERIFICATION PASSED - All checks passed, frames transmitted!'
               else
                 echo '✗ PYTHON STREAMING VERIFICATION FAILED - One or more checks failed'
                 echo ''
                 echo '=== Python Server Log Excerpt ==='
                 head -50 \$SERVER_LOG
                 echo '...'
                 tail -50 \$SERVER_LOG
                 echo ''
                 echo '=== Python Client Log Excerpt ==='
                 head -50 \$CLIENT_LOG
                 echo '...'
                 tail -50 \$CLIENT_LOG
               fi

               # Clean up logs
               rm -f \$SERVER_LOG \$CLIENT_LOG

               # Overall result based on failure flag
               if [ \$TEST_FAILED -eq 0 ]; then
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
