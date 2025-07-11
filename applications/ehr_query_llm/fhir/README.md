# FHIR Client for Retrieving and Posting FHIR Resources

This is an application to interface with a FHIR server to retrieve or post FHIR resources.

It requires that the FHIR Server endpoint URL is provided on the command line as well as client authentication credentials if required. Currently, authentication and authorization is limited to OAuth2.0 server-to-server workflow. When authorization is required by the server, its OAuth2.0 token service URL along with client ID and secret must be provided to the application.

This application also uses ZeroMQ to communicate with its own clients, listening on a well-known port on localhost for messages to retrieve resources of a patient, as well as publishing the retrieved resources on another well-known port. For simplicity, the listening port is defined in the code to be `5600`, and the publishing port `5601`. Messaging security, at transport or message level, is not implemented in this example.

The message schema is simple, with a well-known topic string and topic-specific content schema in JSON format.

The default set of FHIR resource types to retrieve are listed below, which can be overridden by the request message:

- Observation
- ImagingStudy
- FamilyMemberHistory
- Condition
- DiagnosticReport
- DocumentReference

## Requirements

- On a [Holohub supported platform](../../README.md#supported-platforms)
- Python 3.10+
- Python packages from [PyPI](https://pypi.org), including holoscan, fhir.resources, pyzmq, requests and their dependencies

## Run Instructions

There are several ways to build and run this application and package it as a Holoscan Application Package, an [Open Container Initiative](https://opencontainers.org/) compliant image. The following sections describe each in detail.

It is further expected that you have read the [HoloHub README](../../../README.md), have cloned the HoloHub repository to your local system, and the current working directory is the HoloHub root, `holohub`.

**_Note_**:  
The application listens for request messages to start retrieving resources from the server and then publishes the results, so another application is needed to drive this workflow (e.g., the LLM application). To help with simple testing, a Python script is provided as part of this application, and its usage is described below in this [section](#test-the-running-application).

### Quick Start Using Holohub Container

This is the simplest and fastest way to start the application in a Holohub dev container and get it ready to listen to request messages.

**_Note_**:  
Please use your own FHIR server endpoint, as well as the OAuth2.0 authorization endpoint and client credentials as needed.

```bash
./holohub run fhir --run-args "--fhir_url <f_url> --auth_url <a_url> --uid <id> --secret <token>"
```

### Run the Application in Holohub Dev Container

**Launch the container:**

```bash
./holohub run-container fhir
```

This command builds the `holohub:fhir` container based on the application-specific [Dockerfile](./Dockerfile).

**Build and run the application:**

Now in the container, build and run the application:

```bash
~$ pwd
/workspace/holohub

~$ ./holohub clear-cache
~$ ./holohub run fhir --run-args "--fhir_url <f_url> --auth_url <a_url> --uid <id> --secret <token>"
```

Once done, `exit` the container.

### Run the Application in the Host Dev Environment (Bare Metal)

First, create and activate a Python virtual environment, followed by installing the dependencies:

```bash
python3 -m venv .testenv
source .testenv/bin/activate
pip install -r applications/ehr_query_llm/fhir/requirements.txt
```

Then, Set up the Holohub environment:

```bash
./holohub setup  # sudo privileges may be required
```

> Note: Although this application is implemented entirely in Python and relies on standard PyPI packages, you still may want to set up Holohub environment and use `./holohub` commandline .

Next, build and install the application with `./holohub`:

```bash
./holohub install fhir --local
```

Now, run the application which is _installed_ in the `install` folder, with server URLs and credentials of your own:

```bash
python install/bin/fhir/python/fhir_client.py --fhir_url <f_url> --auth_url <a_url> --uid <id> --secret <token>
```

### Test the Running Application

Once the FHIR application has been started with one of the above methods, a test application can be used to request and receive FHIR resources, namely `applications/ehr_query_llm/fhir/test_fhir_client.py`.

The test application contains hard-coded patient name, patient FHIR resource ID, etc., corresponding to a specific test dataset, though it can be easily modified for another dataset.

It is strongly recommended to run this test application in a Python virtual environment, which can be the same as that used for running the FHIR application. The following describes running it in its own environment:

```bash
echo "Assuming venv already created with \`python3 -m venv .testenv\`"
source .testenv/bin/activate
pip install -r applications/ehr_query_llm/fhir/requirements.txt
export PYTHONPATH=${PWD}
python applications/ehr_query_llm/fhir/test_fhir_client.py
```

From the menu, pick one of the choices for the resources of interest.

## Packaging the Application for Distribution and Deployment

With Holoscan CLI, applications built with Holoscan SDK can be packaged into a Holoscan Application Package (HAP), which is an [Open Container Initiative](https://opencontainers.org/) compliant image. An HAP is well suited to be distributed for deployment on hosting platforms, be it Docker Compose, Kubernetes, or otherwise. Please refer to [Packaging Holoscan Applications](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_packager.html) in the User Guide for more information.

This example application provides all the necessary contents for HAP packaging. It is required to perform the packaging in a Python virtual environment, with the application's dependencies installed, before running the following script to reveal specific packaging commands.

```bash
applications/ehr_query_llm/fhir/packageHAP.sh
```

Once the HAP is created, it can then be saved and restored on the target deployment host, and run with the `docker run` command, shown below with user-specific parameters to be substituted.

```bash
docker run -it --rm --net host holohub-fhir-x64-workstation-dgpu-linux-amd64:1.0 \
--fhir_url <f_url> \
--auth_url <a_url> \
--uid <id> \
--secret <token>
```

> **Note:** Packaging this application requires `holoscan-cli`, which can be installed using `pip`. If you are using the same Python environment for packaging as your development environment, there may be a version conflict for the `pydantic` package, as it is required by both `holoscan-cli` and `fhir.resources`. To ensure your development environment can still run the application after packaging, reinstall `fhir.resources`:
>
> ```bash
> pip install fhir.resources
> ```
