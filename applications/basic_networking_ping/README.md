# Basic Networking Ping

This application takes the existing ping example that runs over Holoscan ports and instead uses the basic
network operator to run over a UDP socket.

The basic network operator allows users to send and receive UDP messages over a standard Linux socket.
Separate transmit and receive operators are provided so they can run independently and better suit
the needs of the application.

### Configuration

The application is configured using the file basic_networking_ping_rx.yaml or basic_networking_ping_tx.yaml,
where RX will receive packets and TX will transmit. Depending on how the machine is configured, the IP and
UDP port likely need to be configured. All other settings do not need to be changed.

Please refer to the basic network operator documentation for more configuration information.

### Requirements

This application requires:
1. Linux

### Quick Start

Use the following to build and run the application:

```bash
# Start the receiver
./holohub run basic_networking_ping --language <cpp|python> --run-args basic_networking_ping_rx.yaml
# Start the transmitter
./holohub run basic_networking_ping --language <cpp|python> --run-args basic_networking_ping_tx.yaml
```


## Dev Container

To start the the Dev Container, run the following command from the root directory of Holohub:

```bash
./holohub vscode
```

### VS Code Launch Profiles

#### C++

There are three launch profiles configured for this application:

1. **(gdb) basic_networking_ping/cpp RX**: Launch Basic Networking Ping with the [RX configurations](basic_networking_ping_rx.yaml).
2. **(gdb) basic_networking_ping/cpp TX**: Launch Basic Networking Ping with the [TX configurations](basic_networking_ping_tx.yaml).
3. **(compound) basic_networking_ping/cpp TX & RX**: Launch both 1 and 2 in parallel.
   This launch profile launches the receiver follow by the transmitter.

#### Python

There are several launch profiles configured for this application:

1. **(debugpy) basic_networking_ping/python RX**: Launch Basic Networking Ping with the [RX configurations](basic_networking_ping_rx.yaml).
   This launch profile enables debugging of Python code.
2. **(debugpy) basic_networking_ping/python TX**: Launch Basic Networking Ping with the [TX configurations](basic_networking_ping_tx.yaml).
   This launch profile enables debugging of Python code.
3. **(pythoncpp) basic_networking_ping/python TX**: Launch Basic Networking Ping with the [RX configurations](basic_networking_ping_rx.yaml).
   This launch profile enables debugging of Python and C++ code.
4. **(pythoncpp) basic_networking_ping/python TX**: Launch Basic Networking Ping with the [TX configurations](basic_networking_ping_tx.yaml).
   This launch profile enables debugging of Python and C++ code.
5. **(compound) basic_networking_ping/python TX & RX**: Launch both 1 and 2 in parallel.
   This launch profile launches the receiver follow by the transmitter.
