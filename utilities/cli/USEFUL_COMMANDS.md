# Useful Commands

- To pass options that look like arguments (for example `--run-args`), use `=` instead of a space (see [Python argparse design](https://github.com/python/cpython/issues/53580)):

  ```bash
  --run-args="--verbose"   # instead of --run-args "--verbose"
  ```

- To clear unused Docker cache: `docker image prune`, `docker buildx prune`, or `docker system prune` (see [Docker CLI reference](https://docs.docker.com/reference/cli/docker/)).
- Arguments that used `_` are now `-` (for example `--base_img` → `--base-img`).
- `sudo ./holohub` may not work due to environment filtering (for example `PATH`).
- Running a container as root (for example `--docker-opts="-u root"`) can be necessary for debugging but has security risks; avoid in production.

## Getting Help

```bash
./holohub --help              # General help
./holohub run --help          # Command-specific help
./holohub run myapp --verbose --dryrun  # Debug mode
./holohub list                # Check available projects
```

## Bash Autocompletion

Autocompletion is installed during setup and provides project names, commands, and dynamic discovery. To use it:

```bash
./holohub <TAB><TAB>          # Show all available options
./holohub run ultra<TAB>      # Complete to "ultrasound_segmentation"
```

If autocompletion is not working, you can install it manually:

```bash
sudo cp utilities/holohub_autocomplete /etc/bash_completion.d/
echo ". /etc/bash_completion.d/holohub_autocomplete" >> ~/.bashrc
source ~/.bashrc
```

The autocompletion uses `./holohub autocompletion_list` internally.