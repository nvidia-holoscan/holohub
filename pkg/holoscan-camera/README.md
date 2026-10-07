# holoscan-camera package

This directory defines the Debian package for `holoscan-camera`.

## Usage

```bash
./holoscan_camera package holoscan-camera --pkg-generator DEB
```

The package descriptor does not advertise a language, so omit `--language` when
invoking the package command.

## metadata.json

`metadata.json` registers this package with the holohub CLI. Two fields matter:

- **`package` key** — marks this directory as a HoloHub *package* project. The CLI
  discovers it via the recursive `HOLOSCAN_CLI_SEARCH_PATH` scan from the module root,
  which makes it appear under the `PACKAGES` section of `./holoscan_camera list`.
- **`package.dockerfile`** — declares a Dockerfile path (relative to the module
  root) for this package-project record. When packaging this generated module
  by name, `./holoscan_camera package` instead selects the root `module` record and its
  `module.dockerfile`.

This package is C++ only. Holoscan SDK 5.x EA does not support this module's
Python bindings yet.

## Nested SDK builds

When nested inside the Holoscan SDK, the module installs as component
`holoscan-camera`. Its libraries, headers, executable, and CMake package files
are installed into the parent SDK layout, with the package config and targets
under `<libdir>/cmake/holoscan` as selected by GNUInstallDirs. Set
`HOLOSCAN_CAMERA_BUILD_DEB=ON` to opt into producing the Debian package from a
nested build.
