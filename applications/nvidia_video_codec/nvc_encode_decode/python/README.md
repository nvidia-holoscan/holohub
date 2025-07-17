# NVIDIA Video Codec: Encode-Decode Video

This application demonstrates the use of NVIDIA Video Codec SDK. The application loads a video file, encodes the video using either H.264 or HEVC (H.265), decodes the video, and displays it with Holoviz.

> [!IMPORTANT]  
> By using the NVIDIA Video Codec Demo application and its operators, you agree to the [NVIDIA Software Developer License Agreement](https://developer.nvidia.com/designworks/sdk-samples-tools-software-license-agreement). If you disagree with the EULA, please do not run this application.

### Requirements

- NVIDIA Driver Version >= 570
- CUDA Version >= 12.8
- x86 and SBSA platforms with dedicated GPU

> 💡 **Note:** NVIDIA IGX Orin with integrated GPU is not currently supported.

### Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

## Building and Running the NVIDIA Video Codec Application

### Python

```bash
./holohub run nvc_encode_decode --language python
```

## Configuration

The application is configured with H.264 codec by default. It may be modified in the [nvc_encode_decode.yaml](./nvc_encode_decode.yaml) file:

```yaml
encoder:
  codec: "H264" # H265 or HEVC
  preset: "P3" # P1, P2, P3, P4, P5, P6, P7
  cuda_device_ordinal: 0
  bitrate: 10000000
  frame_rate: 60
  rate_control_mode: 0 # 0: Constant QP, 1: Variable bitrate, 2: Constant bitrate
  multi_pass_encoding: 1 # 0: Disabled, 1: Quarter resolution, 2: Full resolution
```

Refer to the [NVIDIA Video Codec documentation](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-video-encoder-api-prog-guide/) for additional details.

## Benchmarks

We collected latency benchmark results using Holoscan [Data Flow Tracking](https://docs.nvidia.com/holoscan/sdk-user-guide/flow_tracking.html) tools on the NVIDIA Video Codec sample application. The benchmark is conducted on x86_64 with AMD Ryzen 9 7950X, 128 GB system memory and NVIDIA ADA6000 GPU.

**Encoder Configurations:**
- **Bitrate**: 10 MB
- **FPS**: 60
- **Rate Control Mode**: 1 Variable Bitrate
- **Multi-pass Encoding**: 1 Quarter Resolution

<table>
 <thead>
    <tr>
      <th colspan="1"></th>
      <th colspan="1"></th>
      <th colspan="3">E2E</th>
      <th colspan="3">Encoding</th>
      <th colspan="3">Decoding</th>
      <th colspan="1">FPS</th>
    </tr>
    <tr>
      <th scope="col">Codec</th>
      <th scope="col">Preset</th>
      <th scope="col">Min</th>
      <th scope="col">Max</th>
      <th scope="col">Avg</th>
      <th scope="col">Min</th>
      <th scope="col">Max</th>
      <th scope="col">Avg</th>
      <th scope="col">Min</th>
      <th scope="col">Max</th>
      <th scope="col">Avg</th>
      <th scope="col">Avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5">H.264</td>
      <td>P3</td>
      <td>6.242</td>
      <td>8.423</td>
      <td>6.743</td>
      <td>0.536</td>
      <td>0.865</td>
      <td>0.593</td>
      <td>5.258</td>
      <td>7.307</td>
      <td>5.699</td>
      <td>145.270</td>
    </tr>
    <tr>
      <td>P4</td>
      <td>6.22</td>
      <td>8.219</td>
      <td>6.674</td>
      <td>0.561</td>
      <td>0.962</td>
      <td>0.615</td>
      <td>5.220</td>
      <td>7.218</td>
      <td>5.616</td>
      <td>146.875</td>
    </tr>
    <tr>
      <td>P5</td>
      <td>6.508</td>
      <td>8.441</td>
      <td>7.044</td>
      <td>0.921</td>
      <td>1.403</td>
      <td>0.971</td>
      <td>5.229</td>
      <td>7.097</td>
      <td>5.658</td>
      <td>139.433</td>
    </tr>
    <tr>
      <td>P6</td>
      <td>6.37</td>
      <td>9.409</td>
      <td>7.102</td>
      <td>0.680</td>
      <td>1.060</td>
      <td>0.730</td>
      <td>5.141</td>
      <td>7.301</td>
      <td>5.646</td>
      <td>143.368</td>
    </tr>
    <tr>
      <td>P7</td>
      <td>6.529</td>
      <td>8.531</td>
      <td>7.107</td>
      <td>0.740</td>
      <td>1.155</td>
      <td>0.795</td>
      <td>5.104</td>
      <td>8.231</td>
      <td>5.650</td>
      <td>142.727</td>
    </tr>
    <tr>
      <td rowspan="5">HEVC</td>
      <td>P3</td>
      <td>6.258</td>
      <td>9.039</td>
      <td>6.898</td>
      <td>0.684</td>
      <td>1.078</td>
      <td>0.728</td>
      <td>5.141</td>
      <td>7.371</td>
      <td>5.656</td>
      <td>145.058</td>
    </tr>
    <tr>
      <td>P4</td>
      <td>6.146</td>
      <td>9.351</td>
      <td>6.788</td>
      <td>0.684</td>
      <td>1.088</td>
      <td>0.731</td>
      <td>5.130</td>
      <td>7.301</td>
      <td>5.642</td>
      <td>143.481</td>
    </tr>
    <tr>
      <td>P5</td>
      <td>6.193</td>
      <td>8.991</td>
      <td>6.818</td>
      <td>0.680</td>
      <td>1.060</td>
      <td>0.730</td>
      <td>5.130</td>
      <td>7.301</td>
      <td>5.643</td>
      <td>143.371</td>
    </tr>
    <tr>
      <td>P6</td>
      <td>6.337</td>
      <td>9.113</td>
      <td>6.924</td>
      <td>0.682</td>
      <td>1.109</td>
      <td>0.733</td>
      <td>5.254</td>
      <td>8.043</td>
      <td>5.729</td>
      <td>141.633</td>
    </tr>
    <tr>
      <td>P7</td>
      <td>6.246</td>
      <td>10.11</td>
      <td>6.909</td>
      <td>0.740</td>
      <td>1.155</td>
      <td>0.793</td>
      <td>5.104</td>
      <td>8.231</td>
      <td>5.667</td>
      <td>142.035</td>
    </tr>
  </tbody>
</table>

*Note: all reported latency values are in milliseconds.*

## Licensing

Holohub applications and operators are licensed under Apache-2.0.

NVIDIA Video Codec is governed by the terms of the [NVIDIA Software Developer License Agreement](https://developer.nvidia.com/designworks/sdk-samples-tools-software-license-agreement), which you accept by cloning, running, or using the NVIDIA Video Codec sample applications and operators.
