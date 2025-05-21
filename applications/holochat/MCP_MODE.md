# HoloChat MCP Mode

HoloChat now supports running as a Model Context Protocol (MCP) server that provides Holoscan context to upstream LLMs. This allows coding agents and other AI tools to access Holoscan SDK documentation and codebase context via the MCP protocol.

## What is MCP?

The [Model Context Protocol](https://modelcontextprotocol.io/) is a standardized protocol for LLMs to request additional context from external services. By running HoloChat in MCP mode, you provide a context server that can be accessed by compatible LLM clients like Cursor.

## Running HoloChat in MCP Mode

To start HoloChat in MCP mode:

```bash
./dev_container build_and_run holochat --run_args --mcp
```

The server will start on port 8090 (configured in `config.yaml`).

## Using with Cursor

To connect Cursor to your HoloChat MCP server:

1. Add the HoloChat MCP server to your Cursor configuration by editing your `~/.cursor/mcp.json` file (create it if it doesn't exist):

```json
{
  "mcpServers": {
    "holoscan-context": {
      "url": "http://localhost:8090/sse",
      "env": {
        "API_KEY": "value"
      }
    }
  }
}
```

2. If you're running HoloChat on a remote machine (e.g., Ubuntu) and Cursor locally, you'll need to set up port forwarding:

```bash
ssh -L 8090:localhost:8090 username@your-ubuntu-machine
```

3. After configuring the MCP server, you'll need to reopen Cursor or open a new Cursor window for the changes to take effect.

4. Verify the connection by checking the terminal running your HoloChat MCP instance. You should see logs similar to:

```
INFO:     127.0.0.1:51874 - "GET /sse HTTP/1.1" 200 OK
INFO:     127.0.0.1:51878 - "POST /messages/?session_id=c5ce7a2ad956459da06c1582dc6fa14e HTTP/1.1" 202 Accepted
INFO:     127.0.0.1:51900 - "POST /messages/?session_id=c5ce7a2ad956459da06c1582dc6fa14e HTTP/1.1" 202 Accepted
mcp.server.lowlevel.server - INFO - Processing request of type ListToolsRequest
```

These logs indicate that your server is properly connected and your Cursor agent can now use it to get additional Holoscan context.

## MCP Server API

The server exposes a single tool:

### `fetch_context`
Fetches relevant context from the Holoscan database based on a query.

**Input Schema:**
```json
{
  "query": "How to create a Holoscan operator?",
  "num_docs": 5,
  "filter": {"source": {"$contains": "python"}}
}
```

**Parameters:**
- `query` (string, required): The query to search for relevant Holoscan context
- `num_docs` (integer, optional): Number of documents to return (default: 5, max: 15)
- `filter` (object, optional): Filter criteria for the search

## Configuration

MCP server settings can be configured in `config.yaml`:

```yaml
# MCP Server Configuration
mcp_server_name: "holoscan-context-provider"
mcp_server_port: 8090  # Different from local_llm_url port to avoid conflicts
mcp_server_host: "0.0.0.0"

# Number of documents to return from the vector db by default when using MCP
default_num_docs: 10

# Maximum number of documents to return (if requested)
max_num_docs: 30
``` 