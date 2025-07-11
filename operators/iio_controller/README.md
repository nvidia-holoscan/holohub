# IIO Controller Operator

## Overview

The IIO Controller provides a comprehensive set of operators for interfacing with Industrial I/O (IIO) devices in Holoscan applications. These operators enable real-time streaming and control of software-defined radio (SDR) devices, data acquisition systems, and various sensors through the Linux IIO subsystem.

## Description

The IIO Controller operators abstract the complexities of the Linux IIO framework, providing high-performance, low-latency access to:
- **Software Defined Radios (SDRs)** like ADALM-Pluto for RF signal processing
- **Data Acquisition Systems** for high-speed analog/digital conversion
- **Sensors** including accelerometers, gyroscopes, magnetometers, and environmental sensors
- **Signal Generators** and other test equipment

### What is IIO?

The Industrial I/O (IIO) subsystem is a Linux kernel framework that provides:
- Unified API for diverse hardware devices
- High-performance data streaming
- Real-time configuration of device parameters
- Support for triggered sampling and buffered operations

## Requirements

### Software
- libiio (version 0.X)
- Holoscan SDK

## Operator Types

This package provides 5 specialized operators:

### 1. `IIOAttributeRead` - Device Parameter Reading
Reads configuration parameters and real-time status from IIO devices. Use for:
- Monitoring device temperature, gain, frequency settings
- Reading calibration status
- Checking signal strength indicators

### 2. `IIOAttributeWrite` - Device Parameter Control
Writes configuration parameters to IIO devices. Use for:
- Setting RF frequency, gain, bandwidth
- Configuring sampling rates
- Enabling/disabling device features

### 3. `IIOBufferRead` - High-Speed Data Acquisition
Streams data from IIO device buffers with DMA support. Use for:
- Capturing buffer samples from SDRs
- Reading multi-channel ADC data
- Acquiring sensor data streams

### 4. `IIOBufferWrite` - High-Speed Data Transmission
Streams data to IIO device buffers for output. Use for:
- Transmitting buffer samples through SDRs
- Generating analog waveforms via DACs
- Outputting test patterns

### 5. `IIOConfigurator` - Automated Device Setup
Applies complex configurations from YAML files. Use for:
- Initializing devices with multiple parameters
- Switching between operational modes
- Applying calibration profiles

### IIOAttributeRead Operator

#### Configuration Parameters

- **`ctx`**: (Mandatory) The IIO context URI:
  - `"ip:192.168.2.1"` - Network connection (e.g., ADALM-Pluto default)
  - `"usb:3.2.5"` - Direct USB connection
  - `"local:"` - Local IIO devices
  - `"serial:/dev/ttyUSB0,115200"` - Serial connection
- **`dev`**: (Optional) Device name (e.g., `"ad9361-phy"` for Pluto's transceiver)
- **`chan`**: (Optional) Channel name (e.g., `"voltage0"` for RF input)
- **`channel_is_output`**: (Optional) True for TX channels, false for RX channels
- **`attr_name`**: (Mandatory) Attribute to read (e.g., `"frequency"`, `"sampling_frequency"`, `"gain"`).

#### Ports

- To receive the data read from the `IIOAttributeRead` operator, use the
  output port named `value` of type `std::string`.

#### Operator Example - Reading RF Frequency from ADALM-Pluto

```cpp
// Read the current RF frequency from ADALM-Pluto receiver
auto freq_reader = make_operator<ops::IIOAttributeRead>(
    "PlutoFreqReader",
    Arg("ctx") = std::string("ip:192.168.2.1"),  // Pluto's default IP
    Arg("dev") = std::string("ad9361-phy"),      // RF transceiver device
    Arg("chan") = std::string("altvoltage0"),    // RX LO channel
    Arg("channel_is_output") = false,             // Input channel
    Arg("attr_name") = std::string("frequency")   // Read frequency attribute
);

// Connect to a display operator
add_flow(freq_reader, display_op, {{"value", "frequency"}});
```

### IIOAttributeWrite Operator

#### Configuration Parameters

- **`ctx`**: (Mandatory) The URI of the IIO context to connect to the device.
- **`dev`**: (Optional) The name of the IIO device to write to. If not
    specified, it will write to the context attributes.
- **`chan`**: (Optional) The name of the IIO channel to write to. If not
    specified, it will write to the device attributes (the dev parameter must
    be specified then).
- **`channel_is_output`**: (Optional) If true, the channel is treated as an output
    channel. Defaults to false. If the **`chan`** parameter is set, this
    parameter must also be set.
- **`attr_name`**: (Mandatory) The name of the attribute to write to.

#### Ports

- To send the data to be written to the `IIOAttributeWrite` operator, use the
  input port named `value` of type `std::string`.

#### Operator Example

```cpp
auto iio_write_op = make_operator<ops::IIOAttributeWrite>(
    "IIOAttributeWrite",
    Arg("ctx") = std::string("ip:192.168.2.1"),
    Arg("dev") = std::string("ad9361-phy"),
    Arg("chan") = std::string("voltage0"),
    Arg("channel_is_output") = false,
    Arg("attr_name") = std::string("raw")
);

add_flow(basic_emitter_op, iio_write_op, {{"value", "value"}});
```

### IIOBufferRead Operator

#### Configuration Parameters

- **`ctx`**: (Mandatory) IIO context URI (e.g., `"ip:192.168.2.1"` for ADALM-Pluto)
- **`dev`**: (Mandatory) Device name:
  - `"cf-ad9361-lpc"` - Pluto's RX data streaming device
  - Device name for your specific hardware
- **`is_cyclic`**: (Mandatory) Buffer mode:
  - `true` - Continuous streaming (typical for SDR applications)
  - `false` - One-shot capture
- **`samples_count`**: (Mandatory) Samples per channel per buffer (e.g., 8192)
- **`enabled_channel_names`**: (Mandatory) Channel list:
  - `["voltage0", "voltage1"]`
  - `["voltage0"]`
- **`enabled_channel_input`**: (Mandatory) Channel direction list:
  - `[true, true]` - Input channels for RX
  - Must match the order of `enabled_channel_names`

#### Ports

- To receive the data read from the `IIOBufferRead` operator, use the output
  port named `buffer` of type `iio_buffer_info_t` as a shared pointer. This
  structure contains a the sample count and a void pointer to the data.
  This is the data read from the buffer, in order to interpret it, please
  refer to the available example application or the libiio documentation (preferred).

#### Operator Example - Capturing IQ Data from ADALM-Pluto

```cpp
// Configure channels for IQ data reception
std::vector<std::string> rx_channels = {"voltage0", "voltage1"};
std::vector<bool> rx_input_flags = {true, true};  // Both are input channels

// Create SDR receiver operator
auto sdr_receiver = make_operator<ops::IIOBufferRead>(
    "PlutoReceiver",
    Arg("ctx") = std::string("ip:192.168.2.1"),
    Arg("dev") = std::string("cf-ad9361-lpc"),     // RX streaming device
    Arg("is_cyclic") = true,                        // Continuous streaming
    Arg("samples_count") = static_cast<size_t>(8192), // 8K samples per buffer
    Arg("enabled_channel_names") = rx_channels,
    Arg("enabled_channel_input") = rx_input_flags
);

// Connect to signal processing pipeline
add_flow(sdr_receiver, fft_processor, {{"buffer", "iq_data"}});
```

### IIOBufferWrite Operator

#### Configuration Parameters

- **`ctx`**: (Mandatory) IIO context URI
- **`dev`**: (Mandatory) Device name:
  - `"cf-ad9361-dds-core-lpc"` - Pluto's TX data streaming device
- **`is_cyclic`**: (Mandatory) Buffer mode:
  - `true` - Continuous transmission (typical for signal generation)
  - `false` - Single buffer transmission
- **`enabled_channel_names`**: (Mandatory) Output channels:
  - `["voltage0", "voltage1"]`
- **`enabled_channel_output`**: (Mandatory) Channel direction:
  - `[true, true]` - Output channels for TX

#### Ports

- To send the data to be written to the `IIOBufferWrite` operator, use the
  input port named `buffer` of type `iio_buffer_info_t` as a shared pointer.
  This structure contains a the sample count and a void pointer to the data.
  This is the data to be written to the buffer, in order to form it,
  please refer to the available example application or the libiio documentation.

#### Operator Example - Transmitting Data with ADALM-Pluto

```cpp
// Configure TX channels for data transmission
std::vector<std::string> tx_channels = {"voltage0", "voltage1"}; // Two channels
std::vector<bool> tx_output_flags = {true, true};  // Both are output channels

// Create SDR transmitter operator
auto sdr_transmitter = make_operator<ops::IIOBufferWrite>(
    "PlutoTransmitter",
    Arg("ctx") = std::string("ip:192.168.2.1"),
    Arg("dev") = std::string("cf-ad9361-dds-core-lpc"), // TX streaming device
    Arg("is_cyclic") = true,                             // Continuous transmission
    Arg("enabled_channel_names") = tx_channels,
    Arg("enabled_channel_output") = tx_output_flags
);

// Connect data source to transmitter - buffer should contain raw interleaved samples
add_flow(data_generator, sdr_transmitter, {{"output_buffer", "buffer"}});
```

### IIOConfigurator Operator

#### Configuration Parameters

- **`cfg`**: (Mandatory) Path to YAML configuration file

#### YAML Configuration Example for ADALM-Pluto

```yaml
cfg:
  uri: "ip:192.168.2.1"
  setup:
    devices:
      - ad9361-phy:
          attrs:
            - calib_mode: "manual"
            - ensm_mode: "fdd"
          debug-attrs:
            - loopback: 1
      - cf-ad9361-dds-core-lpc:
          channels:
            output:
              - voltage1:
                  attrs:
                    - sampling_frequency: 30719999
      - cf-ad9361-lpc:
          channels:
            input:
              - voltage0:
                  attrs:
                    - sampling_frequency: 30719999
          attrs:
            - sync_start_enable: "arm"
          buffer-attrs:
            - length_align_bytes: 8
```

#### Operator Example

```cpp
// Initialize ADALM-Pluto with complex configuration
auto pluto_config = make_operator<ops::IIOConfigurator>(
    "PlutoConfig",
    Arg("cfg") = std::string("pluto_setup.yaml")
);

// Apply configuration at startup
add_flow(start_op(), pluto_config);
```

## Data Format and Buffer Structure

### Important Notes on Data Handling
The IIO operators provide **direct access to raw device buffers** without any automatic data conversion or interpretation:

1. **No Automatic Data Conversion**: The operators do not automatically interpret channels data (needs conversion in application)
2. **Raw Buffer Access**: Data is passed as-is from/to the IIO device buffers
3. **Application Responsibility**: Your application must handle any necessary data interpretation or conversion
4. **Channel Independence**: Each channel (voltage0, voltage1, etc.) is an independent data stream

### Buffer Memory Layout
When multiple channels are enabled, samples are interleaved in the buffer:
```
Single channel: [Ch0_S0, Ch0_S1, Ch0_S2, ...]
Dual channel:   [Ch0_S0, Ch1_S0, Ch0_S1, Ch1_S1, ...]
```

The exact data format (sample size, endianness, etc.) depends on the specific IIO device configuration.

## Additional Resources

- [Scopy Application](https://github.com/analogdevicesinc/scopy): GUI for IIO devices
- [PyADI-IIO](https://github.com/analogdevicesinc/pyadi-iio): Python bindings for IIO devices
- [Holoscan IIO Examples](../../applications/iio): Complete application examples
