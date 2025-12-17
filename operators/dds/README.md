# Data-Distribution Service (DDS) Operators

This folder contains operators that allow applications to publish or subscribe
to data topics in a DDS domain using [RTI Connext](https://content.rti.com/l/983311/2024-04-30/pz1wms).
These operators demonstrate the ability for Holoscan applications to integrate
and interoperate with applications outside of Holoscan, taking advantage of the
data-centric and distributed nature of DDS to quickly enable communication with
a wide array of external applications and platforms.

## Requirements

- **RTI Connext 7.5.0 Express**  
  - Provides access to the DDS domain.  
  - Already included if using the [container build](../../applications/dds/dds_video/Dockerfile) you can find in the dds_video application directory  
  - Otherwise, install via [RTI APT instructions](https://content.rti.com/l/983311/2025-07-08/q5x1n8).  

- **RTI Activation Key**  
  - [Download the RTI license/activation key](https://content.rti.com/l/983311/2025-07-25/q6729c).  
  - See the [usage rules](https://www.rti.com/products/connext-express).  
  - For Holoscan usage, download the key, copy it into the `holohub` root directory, and rename it to `rti_license.dat`.  

To build on an IGX devkit (using
the `armv8` architecture), follow the
[instructions to build Connext DDS applications for embedded Arm targets](https://community.rti.com/kb/how-do-i-create-connext-dds-application-rti-code-generator-and-build-it-my-embedded-target-arm)
up to step 5 (Installing Java and setting JREHOME).

To build the operators out of the dockerfile, the `RTI_CONNEXT_DDS_DIR` CMake variable must point to
the installation path for RTI Connext. This can be done automatically by setting
the `NDDSHOME` environment variable to the RTI Connext installation directory
(such as when using the RTI `setenv` scripts), or manually at build time, e.g.:

```sh
$ ./holohub build dds_video"
```

## Example Application

See the [DDS Video Application documentation](../../applications/dds/dds_video/README.md) for an example of how to use these operators in a Holoscan application.
