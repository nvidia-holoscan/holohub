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

## Development

### Project Structure

```
{{ cookiecutter.project_slug }}/
├── CMakeLists.txt
├── README.md
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
