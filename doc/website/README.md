# Holohub Website

## Viewing the current HoloHub GitHub Pages

The documentation is viewable at [https://nvidia-holoscan.github.io/holohub](https://nvidia-holoscan.github.io/holohub).

## Previewing the website locally

From the repository root:

```bash
# Build the Docker image
docker build -t holohub-website -f doc/website/Dockerfile .

# Run the Docker container
docker run --rm -it \
    -p 8000:8000 \
    -v ${PWD}:/holohub \
    -w /holohub/doc/website \
    --name holohub-mkdocs \
    holohub-website:latest
```

And then navigate to [`http://0.0.0.0:8000`](http://0.0.0.0:8000) on your local
machine.
