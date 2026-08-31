# Holoscan Connext DDS Module

This in-tree module provides the generic `ConnextDDSPublisherOp` and
`ConnextDDSSubscriberOp` operators. Applications supply their own Python type
from an IDL-generated class or an `rti.idl` declaration.

## Versions

- Module release: `1.0.0` (the module's own semantic version, set in
  `metadata.json` and `pyproject.toml`).
- RTI Connext DDS: `7.7.0`.
- Python API: `rti.connext==7.7.0`.
- Holoscan SDK: `4.5.0` minimum/tested version.

The Connext version is pinned consistently in `Dockerfile`, `pyproject.toml`,
and the operator metadata. The Docker build argument `RTI_CONNEXT_VERSION`
defaults to `7.7.0`; overriding it is only supported when the APT package,
Python package, `NDDSHOME`, and mounted license path are updated together.

For Debian distribution, the package revision should preserve both dependency
versions in its identifier, for example
`holoscan-connext-dds_1.0.0+connext7.7.0-1hsdk4.5.0_arm64.deb`.

## Binary package dependencies

The Python wheel declares and installs the matching Connext Python API:

```text
rti.connext==7.7.0
```

The Debian package cannot depend directly on a Python `pip` distribution. Its
system-package dependencies should instead include the Holoscan runtime and
the RTI Connext DDS 7.7.0 Debian runtime, for example:

```text
holoscan-cuda-13 (= 4.5.0), rti-connext-dds-7.7.0
```

The `rti.connext==7.7.0` wheel must be installed in the Python environment
provided to the application (the module Dockerfile does this for the HoloHub
container). The Debian package must not run `pip` during installation.

Debian package metadata uses vendor `RTI Real Time Innovations` and contact
`juanca@rti.com`. The RTI activation license is proprietary, is not included
in either package, and must be mounted or supplied separately at runtime.

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
