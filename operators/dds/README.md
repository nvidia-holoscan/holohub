# Data-Distribution Service (DDS) Operators

This folder contains operators that allow applications to publish or subscribe
to data topics in a DDS domain using [RTI Connext](https://content.rti.com/l/983311/2024-04-30/pz1wms).
These operators demonstrate the ability for Holoscan applications to integrate
and interoperate with applications outside of Holoscan, taking advantage of the
data-centric and distributed nature of DDS to quickly enable communication with
a wide array of external applications and platforms.

## Requirements

The HoloHub Dockerfile installs RTI Connext DDS **7.7.0** and the native C++
code generator. A valid RTI license is still required at runtime. Download one
from the RTI link used by this repository and place it at
`/opt/rti.com/rti_connext_dds-7.7.0/rti_license.dat` in the container (a
read-only bind mount is recommended):

```sh
curl -fL 'https://content.rti.com/l/983311/2025-07-25/q6729c' -o rti_license.dat
```

Build and run entirely through HoloHub; no host Connext installation is needed:

```sh
./holohub build dds_video --language cpp
./holohub run dds_video --docker-opts \
  "-v $PWD/rti_license.dat:/opt/rti.com/rti_connext_dds-7.7.0/rti_license.dat:ro"
```
