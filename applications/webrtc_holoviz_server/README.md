# WebRTC Holoviz Server

![](screenshot.png)<br>

This app generates video frames with user specified content using Holoviz and sends it to a browser using WebRTC. The goal is to show how to remote control operators and view the output of a Holoscan pipeline.

The app starts a web server, the pipeline starts when a browser is connected to the web server and the `Start` button is pressed. The pipeline stops when the `Stop` button is pressed.

The web page has user inputs for specifying text and for the speed the text moves across the screen.

```mermaid
flowchart LR
    subgraph Server
        GeometryGenerationOp --> HolovizOp
        HolovizOp --> FormatConverterOp
        FormatConverterOp --> WebRTCServerOp
        WebServer
    end
    subgraph Client
        WebServer <--> Browser
        WebRTCServerOp <--> Browser
    end
```

> **_NOTE:_** When using VPN there might be a delay of several seconds between pressing the `Start` button and the first video frames are display. The reason for this is that the STUN server `stun.l.google.com:19302` used by default might not be available when VPN is active and the missing support for Trickle ICE in the used aiortc library. Trickle ICE is an optimization to speed up connection establishment. Normally, possible connections paths are tested one after another. If connections time out this is blocking the whole process. Trickle ICE checks each possible connection path in parallel so the connection timing out won't block the process.

## Prerequisites

The app is using [AIOHTTP](https://docs.aiohttp.org/en/stable/) for the web server and [AIORTC](https://github.com/aiortc/aiortc) for WebRTC. Install both using pip.

```bash
pip install aiohttp aiortc
```

## Run Instructions

Run the command:

```bash
./holohub run webrtc_holoviz_server --local --no-local-build
```

On the same machine open a browser and connect to `127.0.0.1:8080`. You can also connect from a different machine by connecting to the IP address the app is running on.

Press the `Start` button. Video frames are displayed. To stop, press the `Stop` button. Pressing `Start` again will continue the video.

Change the text input and the speed slider to control the generated video frame content.

### Command Line Arguments

```
usage: webrtc_server.py [-h] [--cert-file CERT_FILE] [--key-file KEY_FILE] [--host HOST] [--port PORT] [--verbose VERBOSE] [--ice-server ICE_SERVER]

optional arguments:
  -h, --help            show this help message and exit
  --cert-file CERT_FILE
                        SSL certificate file (for HTTPS)
  --key-file KEY_FILE   SSL key file (for HTTPS)
  --host HOST           Host for HTTP server (default: 0.0.0.0)
  --port PORT           Port for HTTP server (default: 8080)
  --verbose, -v
  --ice-server ICE_SERVER
                        ICE server config in the form of `turn:<ip>:<port>[<username>:<password>]` or `stun:<ip>:<port>`. This option can be specified multiple times to add multiple ICE servers.
```


## Running With TURN server

A TURN server may be needed if you're running in a containerized environment without host networking (e.g. Kubernetes or Docker). Here are some basic steps to run this example with a TURN server.

Run the TURN server in the same machine that you're running the app on.

**Note: It is strongly recommended to run the TURN server with docker network=host for best performance**

```
# This is the external IP address of the machine running the TURN server
export TURN_SERVER_EXTERNAL_IP="<ip>"

# Command below use admin:admin as the username and password as an example
docker run -d --rm --network=host instrumentisto/coturn \
    -n --log-file=stdout \
    --external-ip=$TURN_SERVER_EXTERNAL_IP \
    --listening-ip=$TURN_SERVER_EXTERNAL_IP \
    --lt-cred-mech --fingerprint \
    --user=admin:admin \
    --no-multicast-peers \
    --verbose \
    --realm=default.realm.org
```

Then you can pass in the TURN server config into the app

```
python webrtc_server.py --ice-server "turn:<ip>:3478[admin:admin]"
```

This will enable you to access the webRTC browser application from different machines.
