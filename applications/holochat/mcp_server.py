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

import asyncio
import logging
from typing import List

import mcp.types as types
import uvicorn
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Mount, Route

# Format logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HoloscanContextServer:
    """MCP Server that provides Holoscan context information from a RAG database"""

    def __init__(self, config, db):
        """Initialize the server with the provided configuration and database"""
        self.config = config
        self.server = Server(self.config.mcp_server_name)
        self.db = db

        if not self.db:
            logger.error("Failed to load Chroma database. Please run build_holoscan_db.py first.")
            raise RuntimeError("Chroma database not found")

        # Register the MCP tools
        self._register_tools()

    def _register_tools(self):
        """Register the MCP tools for this server"""

        @self.server.call_tool()
        async def fetch_context(name: str, arguments: dict) -> List[types.TextContent]:
            """Handle tool calls to fetch context from the database"""
            if name != "fetch_context":
                raise ValueError(f"Unknown tool: {name}")

            # Get required argument
            if "query" not in arguments:
                raise ValueError("Missing required argument 'query'")

            query = arguments["query"]

            # Get optional arguments
            num_docs = arguments.get("num_docs", self.config.default_num_docs)
            # Ensure num_docs is within limits
            num_docs = min(num_docs, self.config.max_num_docs)

            # Get filter criteria, if any
            filter_dict = arguments.get("filter", None)

            # Get the most similar documents from the vector db
            docs = self.db.similarity_search_with_score(query=query, k=num_docs, filter=filter_dict)

            # Filter out poor matches from vector db
            docs = list(
                map(
                    lambda lc_doc: lc_doc[0],
                    filter(lambda lc_doc: lc_doc[1] < self.config.search_threshold, docs),
                )
            )

            # Prepare the results
            results = []
            for doc in docs:
                content = doc.page_content
                source = doc.metadata.get("source", "Unknown")

                # Create a formatted document with source information
                formatted_doc = f"Source: {source}\n\n{content}"
                results.append(types.TextContent(type="text", text=formatted_doc))

            return results

        @self.server.list_tools()
        async def list_tools() -> List[types.Tool]:
            """List the available tools for this server"""
            return [
                types.Tool(
                    name="fetch_context",
                    description="Fetches relevant Holoscan documentation and code context based on a query",
                    inputSchema={
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to search for relevant Holoscan context",
                            },
                            "num_docs": {
                                "type": "integer",
                                "description": f"Number of documents to return (default: {self.config.default_num_docs}, max: {self.config.max_num_docs})",
                            },
                            "filter": {
                                "type": "object",
                                "description": "Optional filter criteria for the search (e.g., {'source': {'$contains': 'python'}})",
                            },
                        },
                    },
                )
            ]

    async def run(self, host, port):
        """Run the MCP server using Starlette and uvicorn"""
        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            """Handle the SSE connection for the MCP server"""
            async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
                await self.server.run(
                    streams[0], streams[1], self.server.create_initialization_options()
                )
            return Response()

        async def handle_health(request):
            """Simple health check endpoint"""
            return PlainTextResponse("OK")

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/", endpoint=handle_health, methods=["GET"]),
                Route("/health", endpoint=handle_health, methods=["GET"]),
                Route("/sse", endpoint=handle_sse, methods=["GET"]),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        logger.info(f"Starting Holoscan Context MCP Server on {host}:{port}")
        config = uvicorn.Config(starlette_app, host=host, port=port)
        server = uvicorn.Server(config)
        await server.serve()


def start_mcp_server(config, db):
    """Function to start the MCP server"""
    logger.info("Starting MCP server mode...")

    async def run_server():
        server = HoloscanContextServer(config, db)
        await server.run(config.mcp_server_host, config.mcp_server_port)

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("MCP server stopped by user")
    except Exception as e:
        logger.error(f"Error running MCP server: {e}")
