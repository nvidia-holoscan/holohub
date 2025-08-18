# {{ cookiecutter.project_name }}

{{ cookiecutter.description }}

## Overview

This application is built using Holoscan SDK version {{ cookiecutter.holoscan_version }} and supports the following platforms:
{{ cookiecutter.platforms | replace('[', '') | replace(']', '') }}

## Prerequisites

- Holoscan SDK {{ cookiecutter.holoscan_version }}
- CUDA (if using GPU acceleration)
- Docker (for containerized deployment)

## Installation

1. Clone this repository

2. Install dependencies:

3. Build the application:

## Usage

### Running the Application

```bash
./holohub run {{ cookiecutter.project_slug }}
```

By default, the `./holohub build` and `./holohub run` commands will build and run the application in a containerized environment using the `standard` mode.

For local development without containers, use the `--local` flag:

```bash
./holohub run {{ cookiecutter.project_slug }} --local
```

Note that for the `--local` flag, the relevant custom dependencies (e.g. `requirements.txt` for Python) will be ignored and need to be installed manually.

### For containerized deployment

The application includes a Dockerfile for containerized deployment:

```bash
# Build the container
./holohub build-container {{ cookiecutter.project_slug }}

# Run the containerized application
./holohub run-container {{ cookiecutter.project_slug }}
```

For custom Docker builds:

```bash
# Build with custom base image
./holohub build-container {{ cookiecutter.project_slug }} --base-image nvcr.io/nvidia/clara-holoscan/holoscan:v{{ cookiecutter.holoscan_version }}-dgpu

# Run with specific GPU type
./holohub run-container {{ cookiecutter.project_slug }} --gpu-type dgpu
```

## Development

### Project Structure

```
{{ cookiecutter.project_slug }}/
├── CMakeLists.txt
├── Dockerfile
├── README.md
{% if cookiecutter.language == "python" %}├── requirements.txt{% endif %}
├── src/
│   └── main.{{ 'py' if cookiecutter.language == 'python' else 'cpp' }}
├── include/
├── tests/
└── docs/
```

### Adding New Operators

1. Create a new operator class in `src/operators/`
2. Include the operator in `src/main.{{ 'py' if cookiecutter.language == 'python' else 'cpp' }}`
3. Update the pipeline configuration

## License

This project is licensed under the {{ cookiecutter.license }} License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Authors

- {{ cookiecutter.full_name }} - {{ cookiecutter.affiliation }}

## Acknowledgments

- NVIDIA Holoscan Team
- Open source community contributors
