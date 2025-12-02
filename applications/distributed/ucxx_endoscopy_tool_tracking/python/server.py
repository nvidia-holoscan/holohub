"""
SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # no qa

import argparse
import logging
import os
import sys
import time
import asyncio
import threading
import ucxx
import cupy as cp

from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import FormatConverterOp, HolovizOp
from holoscan.resources import BlockMemoryPool, CudaStreamPool, MemoryStorageType
from holoscan.resources import UcxEntitySerializer

from holoscan.conditions import PeriodicCondition


# Server listens on all interfaces, client connects to localhost
SERVER_HOST = "0.0.0.0"  # Listen on all interfaces
CLIENT_HOST = "127.0.0.1"  # Connect to localhost (or use actual IP for remote connections)
PORT_NUMBER = 13338


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ucxx_endoscopy_tool_tracking_server")

class UCXXSenderOp(Operator):
    serializer: UcxEntitySerializer = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.port = PORT_NUMBER
        self.n_bytes = 2**30

        self.host = CLIENT_HOST  # Client connects to this address
        self.msg = cp.zeros(self.n_bytes, dtype="u1")  # create some data to send
        
        # Create event loop for async operations
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"UCXX Sender initialized, will connect to {self.host}:{self.port}")

    def setup(self, spec: OperatorSpec):
        pass

    def compute(self, op_input, op_output, context):
        try:
            self.logger.info(f"Attempting to connect and send to {self.host}:{self.port}")
            ep = self.loop.run_until_complete(ucxx.create_endpoint(self.host, self.port))
            self.loop.run_until_complete(ep.send(self.msg))
            self.logger.info("Message sent successfully")
            self.loop.run_until_complete(ep.close())
        except (asyncio.TimeoutError, TimeoutError):
            self.logger.warning("Connection/send timeout - server may not be available")
        except Exception as e:
            self.logger.warning(f"Error: {e}")
    
    def stop(self):
        """Clean up resources when operator stops"""
        if self.loop:
            self.loop.close()
        self.logger.info("UCXX Sender closed")


class UCXXReceiverOp(Operator):
    listener = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.port = PORT_NUMBER
        self.n_bytes = 2**30

        self.host = SERVER_HOST  # Server listens on this address
        
        # Queue to store received data from listener callback
        self.data_queue = asyncio.Queue()
        
        # Create event loop for async operations
        self.loop = asyncio.new_event_loop()
        
        # Flag to control the event loop thread
        self._running = True
        
        # Start event loop in background thread
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()
        
        # Create listener with callback (must be done after loop is running)
        time.sleep(0.1)  # Give loop time to start
        asyncio.run_coroutine_threadsafe(self._create_listener(), self.loop).result()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"UCXX Listener created on {self.host}:{self.port}")
    
    def _run_event_loop(self):
        """Run event loop in background thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    async def _create_listener(self):
        """Create the listener in the event loop"""
        self.listener = ucxx.create_listener(self._handle_connection, self.port)

    def _handle_connection(self, ep):
        """Callback for when a client connects - handles recv/send and stores result"""
        self.logger.info("New connection received")
        # Create a task for processing this endpoint
        return asyncio.create_task(self._process_endpoint(ep))

    async def _process_endpoint(self, ep):
        """Async function to handle receiving and sending data"""
        try:
            # Receive buffer
            arr = cp.empty(self.n_bytes, dtype="u1")
            await ep.recv(arr)
            assert cp.count_nonzero(arr) == cp.array(0, dtype=cp.int64)
            self.logger.info("Received CuPy array")
            
            # Store received data in queue for compute to process
            await self.data_queue.put(arr)
        except asyncio.CancelledError:
            self.logger.warning("Endpoint processing cancelled")
        except Exception as e:
            self.logger.error(f"Error processing endpoint: {e}", exc_info=True)

    def setup(self, spec: OperatorSpec):
        pass

    def compute(self, op_input, op_output, context):
        # Poll for available messages without blocking
        try:
            # Check if there's received data available
            if not self.data_queue.empty():
                # Get the data using threadsafe coroutine call
                future = asyncio.run_coroutine_threadsafe(self.data_queue.get(), self.loop)
                received_data = future.result(timeout=0.1)
                self.logger.info(f"Processing received data in compute, shape: {received_data.shape}")
        except asyncio.TimeoutError:
            # No message available yet - this is normal
            pass
        except asyncio.CancelledError:
            # Task was cancelled - this can happen during shutdown
            self.logger.debug("Compute cancelled")
        except Exception as e:
            self.logger.error(f"Error in compute: {e}", exc_info=True)
    
    def stop(self):
        """Clean up resources when operator stops"""
        try:
            if self.listener:
                self.listener.close()
        except Exception as e:
            self.logger.warning(f"Error closing listener: {e}")
        
        try:
            # Stop the event loop
            self._running = False
            if self.loop and not self.loop.is_closed():
                self.loop.call_soon_threadsafe(self.loop.stop)
                self.loop_thread.join(timeout=2.0)
        except Exception as e:
            self.logger.warning(f"Error stopping event loop: {e}")
        
        self.logger.info("UCXX Listener closed")
        

class UCXXEndoscopyToolTrackingServer(Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compose(self):
        receiver_op = UCXXReceiverOp(self, "receiver_op",
                                 condition=PeriodicCondition(self, 0.5))
        self.add_operator(receiver_op)

class UCXXEndoscopyToolTrackingClient(Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compose(self):
        sender_op = UCXXSenderOp(self, "sender_op", 
                                 condition=PeriodicCondition(self, 1.0))  # Tick once per second
        self.add_operator(sender_op)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCXX Endoscopy Tool Tracking Server")
    parser.add_argument("--mode", type=str, default="server", choices=["server", "client"])
    args = parser.parse_args()

    if args.mode == "server":
        server = UCXXEndoscopyToolTrackingServer()
        server.run()
    elif args.mode == "client":
        client = UCXXEndoscopyToolTrackingClient()
        client.run()
