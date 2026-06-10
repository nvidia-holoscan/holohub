# DAQIRI Socket Ping

This application takes the existing ping example that runs over Holoscan ports and instead uses DAQIRI
socket transport to run over a UDP socket. The transmit and receive operators run in one process using
DAQIRI client and server socket interfaces.

## Configuration

The application is configured using the file [daqiri_socket_ping.yaml](daqiri_socket_ping.yaml).
Depending on how the machine is configured, the IP and UDP port likely need to be configured. All other
settings do not need to be changed.

Please refer to [DAQIRI documentation](https://github.com/NVIDIA/daqiri#documentation) for more configuration information.

## Requirements

The HoloHub networking container builds DAQIRI from
[`NVIDIA/daqiri`](https://github.com/NVIDIA/daqiri), installs it as the `daqiri`
system package in the image, and exposes it from `/opt/daqiri`. To use a specific
DAQIRI revision, pass a Docker build argument such as
`--build-args "--build-arg DAQIRI_REF=<tag-or-sha>"`.

## Quick Start

Use the following to build and run the application:

```bash
./holohub run daqiri_socket_ping
```

For using different language implementations, use the `--language` argument, for instance:

```bash
./holohub run daqiri_socket_ping --language cpp
./holohub run daqiri_socket_ping --language python
```

For using different configuration files, use the `--run-args` argument.

```bash
./holohub run daqiri_socket_ping --run-args daqiri_socket_ping.yaml
```
