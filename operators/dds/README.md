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
any application using one of these operators. To build on an IGX devkit (using
the `armv8` architecture), follow the
[instructions to build Connext DDS applications for embedded Arm targets](https://community.rti.com/kb/how-do-i-create-connext-dds-application-rti-code-generator-and-build-it-my-embedded-target-arm)
up to step 5 (Installing Java and setting JREHOME).

To build the operators, the `RTI_CONNEXT_DDS_DIR` CMake variable must point to
the installation path for RTI Connext. This can be done automatically by setting
the `NDDSHOME` environment variable to the RTI Connext installation directory
(such as when using the RTI `setenv` scripts), or manually at build time, e.g.:

```sh
$ ./run build dds_video --configure-args -DRTI_CONNEXT_DDS_DIR=~/rti/rti_connext_dds-6.1.2
```

##### Using a Development Container

Due to the license requirements of RTI Connext it is not currently supported to
install RTI Connext into a development container. Instead, if a development
container is to be used, Connext should be installed onto the host as above and
then the container can be launched with the RTI Connext folder mounted at
runtime. To do so, ensure that the `NDDSHOME` environment variable is set and
use the following:

```sh
./dev_container launch --docker_opts "-v $NDDSHOME:/opt/dds -e NDDSHOME=/opt/dds"
```
