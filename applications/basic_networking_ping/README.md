# Basic Networking Ping

This application takes the existing ping example that runs over Holoscan ports and instead uses DAQIRI
socket transport to run over a UDP socket. The transmit and receive operators run in one process using
DAQIRI client and server socket interfaces.

## Configuration

The application is configured using the file [basic_networking_ping.yaml](basic_networking_ping.yaml).
Depending on how the machine is configured, the IP and UDP port likely need to be configured. All other
settings do not need to be changed.

Please refer to DAQIRI documentation for more configuration information.

## Quick Start

Use the following to build and run the application:

```bash
./holohub run basic_networking_ping
```

For using different language implementations, use the `--language` argument, for instance:

```bash
./holohub run basic_networking_ping --language cpp
./holohub run basic_networking_ping --language python
```

For using different configuration files, use the `--run-args` argument.

```bash
./holohub run basic_networking_ping --run-args basic_networking_ping.yaml
```
