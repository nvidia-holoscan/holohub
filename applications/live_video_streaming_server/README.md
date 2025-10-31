# Live Video Streaming Server

This application provides a **server-only** WebRTC video streaming service that captures video from a camera and streams it to web browsers via WebRTC. External web applications can connect to the streaming API to receive the video feed.

## Architecture

```mermaid
flowchart LR
    subgraph Server
        A[Camera/V4L2] --> B[V4L2VideoCaptureOp]
        B --> C[FormatConverterOp]
        C --> D[WebRTCServerOp]
        E[HTTP API Server]
    end
    subgraph Client
        F[External Web App] <--> E
        D <--> F
        G[Video Element] <-- F
    end
```

## What This Provides

-  **WebRTC Streaming API**: RESTful endpoints for WebRTC signaling
-  **Camera Capture**: Real-time video from `/dev/video0`
-  **CORS Support**: Cross-origin requests for external web applications
-  **Example Client**: Complete HTML5 client implementation
-  **No Built-in Web UI**: Server-only (no embedded client interface)

## Quick Start

### 1. Build the Application

```bash
# Build the application first
./holohub build live_video_streaming_server
```

### 2. Start the Server

```bash
# Run the built application
./holohub run live_video_streaming_server

# Or run directly with Python
python live_video_streaming_server.py

# With custom configuration
python live_video_streaming_server.py --host 0.0.0.0 --port 8080 --verbose
```

### 3. Use the Example Client

First, serve the HTML file through an HTTP server to avoid CORS issues:

```bash
# Navigate to the live_video_streaming_server directory
cd applications/live_video_streaming_server

# Option 1: Using Python (recommended)
python -m http.server 3000

# Option 2: Using Node.js (if available)
npx http-server -p 3000

```

Then open your web browser and navigate to:
- `http://localhost:3000/example_client.html`
- Configure server URL (default: `http://localhost:8080`)
- Click "Connect to Stream" to start receiving video
- Video will appear in the browser once connected

## API Endpoints

The server provides a RESTful API for WebRTC integration:

### `GET /iceServers`
Returns ICE server configuration for WebRTC connection establishment.

**Response:**
```json
[
  {"urls": "stun:stun.l.google.com:19302"},
  {"urls": "turn:10.0.0.131:3478", "username": "admin", "credential": "admin"}
]
```

### `POST /offer`
Handles WebRTC signaling (SDP offer/answer exchange).

**Request:**
```json
{
  "sdp": "v=0\r\no=- 123456789...",
  "type": "offer"
}
```

**Response:**
```json
{
  "sdp": "v=0\r\no=- 987654321...",
  "type": "answer"
}
```

## Command Line Options

```bash
python live_video_streaming_server.py [OPTIONS]

Options:
  --host HOST                    Host for HTTP server (default: 0.0.0.0)
  --port PORT                    Port for HTTP server (default: 8080)
  --cert-file CERT_FILE          SSL certificate file (for HTTPS)
  --key-file KEY_FILE            SSL key file (for HTTPS)
  --ice-server ICE_SERVER        ICE server config: stun:host:port or turn:host:port[user:pass]
  --verbose, -v                  Enable debug logging
  -h, --help                     Show help message
```

## Advanced Configuration

### With TURN Server (for NAT traversal)

```bash
# Set up TURN server (example with Docker)
export TURN_SERVER_EXTERNAL_IP="<your-ip>"
docker run -d --rm --network=host instrumentisto/coturn \
    -n --log-file=stdout \
    --external-ip=$TURN_SERVER_EXTERNAL_IP \
    --listening-ip=$TURN_SERVER_EXTERNAL_IP \
    --lt-cred-mech --fingerprint \
    --user=admin:admin \
    --no-multicast-peers \
    --verbose \
    --realm=default.realm.org

# Run server with TURN configuration
python live_video_streaming_server.py --ice-server "turn:<ip>:3478[admin:admin]"
```

### With HTTPS/SSL

```bash
python live_video_streaming_server.py \
  --cert-file /path/to/cert.pem \
  --key-file /path/to/key.pem
```

## Integration Example

Basic JavaScript integration for custom web applications:

```javascript
// 1. Get ICE servers from the streaming server
const iceServers = await fetch('http://your-server:8080/iceServers')
  .then(response => response.json());

// 2. Create WebRTC peer connection
const pc = new RTCPeerConnection({
  sdpSemantics: 'unified-plan',
  iceServers: iceServers
});

// 3. Handle incoming video stream
pc.addEventListener('track', function(evt) {
  if (evt.track.kind === 'video') {
    document.getElementById('video').srcObject = evt.streams[0];
  }
});

// 4. Add video transceiver (receive only)
pc.addTransceiver('video', { direction: 'recvonly' });

// 5. Create offer and send to server
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

const response = await fetch('http://your-server:8080/offer', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ sdp: offer.sdp, type: offer.type })
});

// 6. Set server's answer
const answer = await response.json();
await pc.setRemoteDescription(answer);
```

## Use Cases

- **Microservice Architecture**: WebRTC server as a dedicated streaming service
- **Multiple Clients**: Different web apps connecting to the same video stream
- **Custom Interfaces**: Build your own UI with authentication, user management, etc.
- **System Integration**: Embed video streaming into existing applications

## Troubleshooting

### Common Issues

1. **Camera Access**: Ensure `/dev/video0` exists and is accessible
   ```bash
   ls -la /dev/video*
   # Should show your camera device
   ```

2. **Network Connectivity**: When using VPN, there might be delays due to STUN server accessibility and lack of Trickle ICE support

3. **CORS Issues**: The server includes CORS headers for cross-origin requests

### VPN Considerations

> **Note:** When using VPN, there might be a delay of several seconds seeing video frames.

## Files

- `live_video_streaming_server.py` - Main server application
- `example_client.html` - Complete HTML5 client example
- `README.md` - Detailed server documentation
