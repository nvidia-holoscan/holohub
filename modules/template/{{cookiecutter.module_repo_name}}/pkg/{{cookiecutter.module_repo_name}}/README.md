# {{ cookiecutter.module_repo_name }} package

This directory defines the Debian package for `{{ cookiecutter.module_repo_name }}`.

## Usage

```bash
./holohub package {{ cookiecutter.module_repo_name }} --pkg-generator DEB
```

## metadata.json

`metadata.json` registers this package with the holohub CLI. Two fields matter:

- **`package` key** — marks this directory as a HoloHub *package* project. The CLI
  discovers it via the recursive `HOLOSCAN_CLI_SEARCH_PATH` scan from the module root,
  which makes it appear under the `PACKAGES` section of `./holohub list`.
- **`package.dockerfile`** — declares a Dockerfile path (relative to the module
  root) for this package-project record. When packaging this generated module
  by name, `./holohub package` instead selects the root `module` record and its
  `module.dockerfile`.

Looking for Python packaging? Review the project [pyproject.toml](../../pyproject.toml)
for configuration.
