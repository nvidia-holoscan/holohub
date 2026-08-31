# Holoscan Connext DDS Module

This in-tree module provides the generic `ConnextDDSPublisherOp` and
`ConnextDDSSubscriberOp` operators. Applications supply their own Python type
from an IDL-generated class or an `rti.idl` declaration.

The module is built from source inside the HoloHub container workflow, which
exposes the operators as `holohub.connext_dds`. This in-tree integration does
not provide standalone Python or Debian packages.

## Versions

- Module release: `1.0.0` (the module's own semantic version, set in
  `metadata.json`).
- RTI Connext DDS: `7.7.0`.
- Python API: `rti.connext==7.7.0`.
- Holoscan SDK: `4.5.0` minimum/tested version.

The Connext version is pinned in `Dockerfile` and the module and operator
metadata. The Docker build argument `RTI_CONNEXT_VERSION`
defaults to `7.7.0`; overriding it is only supported when the APT package,
Python package, `NDDSHOME`, and mounted license path are updated together.

## Container dependencies

The module Dockerfile extends the Holoscan SDK image and installs both the
`rti-connext-dds-7.7.0` package from RTI's APT repository and the matching
`rti.connext==7.7.0` Python API. The APT package alone does not install the
Python API used by these operators. All dependencies stay inside the container.

The RTI activation license is proprietary, is not included in the image, and
must be mounted separately at runtime.

Maintainer: RTI Real Time Innovations. Contact: `juanca@rti.com`.

## Run the example in a container

The recommended workflow is container-only: it keeps the host Python
environment clean and avoids installing RTI or Holoscan dependencies on the
IGX host. From the HoloHub repository root:

```bash
# Download the RTI evaluation license (do not commit this ignored file).
curl -L https://content.rti.com/l/983311/2025-07-25/q6729c -o rti_license.dat

# Build the module image through HoloHub/Docker.
./holohub build-container connext_dds

# Build the example application through HoloHub/Docker.
./holohub build connext_dds_example --language python

# Run the example, mounting the license read-only into the container.
./holohub run connext_dds_example --language python \
  --docker-opts="-v $(pwd)/rti_license.dat:/opt/rti.com/rti_connext_dds-7.7.0/rti_license.dat:ro"
```

The example declares its DDS type with `rti.idl`, publishes 20 samples, and
checks the publisher/subscriber round trip. A successful run ends with a
message such as `DDS round trip succeeded with 19 samples`.

Direct local installation is intentionally not the documented path. If a
consumer chooses to run outside a container, they must reproduce the image's
Holoscan SDK, RTI Connext 7.7.0, Python, license, and environment setup
themselves; the container workflow remains the reference configuration.

RTI Connext DDS 7.7.0 is installed by the module Dockerfile. A valid RTI
activation license must be supplied at runtime; it is not distributed with
HoloHub.
