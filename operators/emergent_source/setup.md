# Emergent Vision Technologies Setup

Thanks to a collaboration with [Emergent Vision Technologies](https://emergentvisiontec.com/), HoloHub provides support for EVT high-speed cameras on NVIDIA Developer Kits equipped with a [ConnectX NIC](https://www.nvidia.com/en-us/networking/ethernet-adapters/) using the [Rivermax SDK](https://developer.nvidia.com/networking/rivermax) through the [Emergent Source operator](README.md).

## Installing EVT Hardware

The EVT cameras can be connected to NVIDIA Developer Kits through a [Mellanox ConnectX SmartNIC](https://www.nvidia.com/en-us/networking/ethernet-adapters/), with the most simple connection method being a single cable between a camera and the devkit.
For 25 GigE cameras that use the SFP28 interface, this can be achieved by using [SFP28](https://store.nvidia.com/en-us/networking/store/product/MCP2M00-A001E30N/NVIDIAMCP2M00A001E30NDACCableEthernet25GbESFP281m/) cable with [QSFP28 to SFP28 adaptor](https://store.nvidia.com/en-us/networking/store/product/MAM1Q00A-QSA28/NVIDIAMAM1Q00AQSA28CableAdapter100Gbsto25GbsQSFP28toSFP28/).

> **Note:**
>
> The HoloHub applications have been tested using a SFP28 copper cable of 2M or
> less. Longer copper cables or optical cables and optical modules can be used but
> these have not been tested as a part of this development.

Refer to the [NVIDIA IGX Orin Developer Kit User Guide](https://developer.nvidia.com/igx-orin-developer-kit-user-guide)
for the location of the QSFP28 connector on the device.

For EVT camera setup, refer to Hardware Installation in EVT [Camera User's Manual](https://emergentvisiontec.com/resources/?tab=umg). Users need to log in to be
able to download Camera User's Manual.

> **Tip:**
>
> The EVT cameras require the user to buy the lens. Based on the application of camera,
> the lens can be bought from any [online](https://www.bhphotovideo.com/c/search?Ntt=c%20mount%20lens&N=0&InitialSearch=yes&sts=ps) store.

## Installing EVT Software

The Emergent SDK needs to be installed in order to compile and run the
HoloHub applications with EVT camera. The latest tested version of the Emergent SDK is `eSDK 2.37.05 Linux Ubuntu 20.04.04 Kernel 5.10.65 JP 5.0 HP`
and can be downloaded from [here](https://emergentvisiontec.com/resources/?tab=ss).
The Emergent SDK comes with headers, libraries and examples. To install the SDK
refer to the Software Installation section of EVT [Camera User's Manual](https://emergentvisiontec.com/resources/?tab=umg). Users need to log in to be
able to download Camera User's Manual.

> **Note:**
>
> The Emergent SDK depends on Rivermax SDK and the Mellanox OFED Network Drivers. If they're already installed on your system, use the following command when installing the Emergent SDK to avoid duplicate installation:
>
> ```bash
> sudo ./install_eSdk.sh no_mellanox
> ```
>
> Ensure the [ConnectX is properly configured](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/additional-setup/enabling-gpu-direct-rdma#enabling-rdma-on-the-connectx-smartnic) to use it with the Emergent SDK.

## Testing the EVT Camera

To test if the EVT camera and SDK was installed correctly, run the `eCapture`
application with `sudo` privileges.  First, ensure that a valid Rivermax license
file is under `/opt/mellanox/rivermax/rivermax.lic`, then follow the instructions
under the eCapture section of EVT [Camera User's Manual](https://emergentvisiontec.com/resources/?tab=umg).

## Troubleshooting

1. **Problem:** The application fails to find the EVT camera.

   **Solution:**

   - Make sure that the MLNX ConnectX SmartNIC is configured with the correct IP address. Follow
     section [Configure the ConnectX SmartNIC](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/additional-setup/enabling-gpu-direct-rdma#enabling-rdma-on-the-connectx-smartnic)

2. **Problem:** The application fails to open the EVT camera.

   **Solutions:**

   - Make sure that the application was run with `sudo` privileges.
   - Make sure a valid Rivermax license file is located at `/opt/mellanox/rivermax/rivermax.lic`.

3. **Problem:** Fail to find `eCapture` application in the home window.

   **Solution:**

   - Open the terminal and find it under `/opt/EVT/eCapture`. The applications needs to
     be run with `sudo` privileges.

4. **Problem:** The `eCapture` application fails to connect to the EVT camera with
   error message "GVCP ack error".

   **Solutions:**

   - It could be an issue with the HR12 power connection to the camera. Disconnect the
     HR12 power connector from the camera and try reconnecting it.

5. **Problem:** The IP address of the Emergent camera is reset even after setting up with the above steps.

   **Solutions:**

   - Check whether the NIC settings in Ubuntu is set to "Connect automatically". Go to `Settings`->`Network`->`NIC for the Camera` and then unselect "Connect automatically" and in the IPv6 tab, select `Disable`.
