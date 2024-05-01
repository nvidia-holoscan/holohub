### DDS Operators

This folder contains operators that allow applications to publish or subscribe
to data topics in a DDS domain using [RTI Connext](https://content.rti.com/l/983311/2024-04-30/pz1wms).
These operators demonstrate the ability for Holoscan applications to integrate
and interoperate with applications outside of Holoscan, taking advantage of the
data-centric and distributed nature of DDS to quickly enable communication with
a wide array of external applications and platforms.

#### Requirements

[RTI Connext](https://content.rti.com/l/983311/2024-04-30/pz1wms) must be
installed on the system and a valid RTI Connext license must be installed to run
any application using one of these operators.

To build the operators, the `RTI_CONNEXT_DDS_DIR` CMake variable must point to
the installation path for RTI Connext. This can be done automatically by setting
the `NDDSHOME` environment variable to the RTI Connext installation directory
(such as when using the RTI `setenv` scripts), or manually at build time, e.g.:

```sh
$ ./run build dds_video --configure-args -DRTI_CONNEXT_DDS_DIR=~/rti/rti_connext_dds-6.1.2
```
