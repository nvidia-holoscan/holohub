### DDS Operators

This folder contains operators that allow applications to publish or subscribe
to data topics in a DDS domain using [RTI Connext](https://www.rti.com/products).
These operators demonstrate the ability for Holoscan applications to integrate
and interoperate with applications outside of Holoscan, taking advantage of the
data-centric and distributed nature of DDS to quickly enable communication with
a wide array of external applications and platforms.

#### Requirements

[RTI Connext](https://www.rti.com/products) must be installed on the system and
a valid RTI Connext license must be installed to run any application using one
of these operators.

The `RTI_CONNEXT_DDS_DIR` variable must be set at build time to specify the
installation path for RTI Connext. For example,

```sh
$ ./run build dds_video --configure-args -DRTI_CONNEXT_DDS_DIR=~/rti/rti_connext_dds-6.1.2
```
