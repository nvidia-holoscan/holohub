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
./dev_container build_and_run basic_networking_ping --base_img holoscan-dev-container:main --language <cpp|python> --run_args basic_networking_ping_rx.yaml
# Start the transmitter
./dev_container build_and_run basic_networking_ping --base_img holoscan-dev-container:main --language <cpp|python> --run_args basic_networking_ping_tx.yaml
```


### Build Instructions

Please refer to the top level Holohub README.md file for information on how to build this application.

### Run Instructions

Running the sample uses the standard HoloHub `run` script:


```bash
# Start the receiver
./run launch basic_networking_ping <language> --extra_args basic_networking_ping_rx.yaml
# Start the transmitter
./run launch basic_networking_ping <language> --extra_args basic_networking_ping_tx.yaml
```

Note: `language`  can be either C++ or Python.
