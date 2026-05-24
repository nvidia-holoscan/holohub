# Basic Networking Ping

This application takes the existing ping example that runs over Holoscan ports and instead uses the basic
network operator to run over a UDP socket.

The basic network operator allows users to send and receive UDP messages over a standard Linux socket.
Separate transmit and receive operators are provided so they can run independently and better suit
the needs of the application.

## Configuration

The application is configured using the file basic_networking_ping_rx.yaml or basic_networking_ping_tx.yaml,
where RX will receive packets and TX will transmit. Depending on how the machine is configured, the IP and
UDP port likely need to be configured. All other settings do not need to be changed.

Please refer to the basic network operator documentation for more configuration information.

## Quick Start

Use the following to build and run the application:

```bash
# Start the receiver
./holohub run basic_networking_ping rx
# Start the transmitter
./holohub run basic_networking_ping tx
```

For using different language implementations, use the `--language` argument, for instance:

```bash
./holohub run basic_networking_ping rx --language cpp
./holohub run basic_networking_ping tx --language python
```

For using different configuration files, use the `--run-args` argument.

```bash
./holohub run basic_networking_ping --run-args basic_networking_ping_rx.yaml
```
