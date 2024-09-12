# FHIR Client for Retrieving FHIR Resources for Given Patient

This application is a client of a FHIR server for retrieving FHIR resources for a given patient.

It requires the FHIR Server endpoint URL be provided on the command line. Client authentication and authorization is also supported, though limited to OAuth2.0 server to server workflow, and when authorization is required by the server, OAuth2.0 token services URL along with client ID and secret need to be provided as command line options.

This application uses ZeroMQ to communicate with its own clients, listening on a wellknown port for request message to retrieve resources of a patient, and sending the retrieved resourced in a message to a destination port on the local host.

## Requirements

- On a [Holohub supported platform](../../README.md#supported-platforms)
- Python 3.10+
- Python packages on [Pypi](https://pypi.org), including holoscan, fhir.resources, holoscan, pyzmq, requests and their dependencies

## Run Instructions

### Quick Start Using Holohub Container


### Run the Application in Dev Environment

### Run the Application in Dev Container

## Packaging the Application for Distribution

With Holoscan CLI, an applications built with Holoscan SDK can be packaged into a Holoscan Application Package (HAP), which is essentially a Open Container Initiative compliant container image. An HAP is well suited to be distributed for deployment on hosting platforms, be a Docker Compose, Kubernetes, or else. Please refer to [Packaging Holoscan Applications](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_packager.html) in the User Guide for more information.

This example application provides all the necessary contents for HAP packaging, and the specific commands are revealed by the specific commands.
