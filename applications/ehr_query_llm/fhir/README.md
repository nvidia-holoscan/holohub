# FHIR Client Application for Retrieving and posting FHIR Resources

This is an application to interface with a FHIR server to retrieve or post FHIR resources.

It requires the FHIR Server endpoint URL be provided on the command line as well as client authentication credentials if required. As of now, authentication and authorization is limited to OAuth2.0 server to server workflow. When authorization is requiired by the server, its OAuth2.0 token service URL along with client ID and secret must be provided to the application.

This application also uses ZeroMQ to communicate with its own clients, listening on a well known port on local host for messages to retrieve resources of a patient, as well as publishing the retrieved resources on another well known port. For simplicity, the listening port is defined in the code to be `5600`, and the publishing port `5601`. Messaging security, at transport or message level, is not implemented in this example.

Message schema is simple, with a well known topic string and topic specific content schema in JSON format.

The default set of FHIR resource types to retrieve are listed below, which can be overridden by the request message
- Observation
- ImagingStudy
- FamilyMemberHistory
- Condition
- DiagnosticReport
- DocumentReference

## Requirements

- On a [Holohub supported platform](../../README.md#supported-platforms)
- Python 3.10+
- Python packages on [Pypi](https://pypi.org), including holoscan, fhir.resources, holoscan, pyzmq, requests and their dependencies

## Run Instructions

There are several ways to build and run this application and package it as a Holoscan Application Package, an [Open Container Initiative](https://opencontainers.org/) compliant image. The following sections describe each in detail.

It is further expected that you have read the [HoloHub README](../../../README.md), have cloned the HoloHub repository to your local system, and the current working directory is the HoloHub root, `holohub`.

**_Note_**:
The application listens on request message to start retrieving resources from the server and then publishes the results, so another application is needed to drive this workflow, e.g. the LLM application. To help with simple testing, a Python script is provided as part of this application, and its usage is described below in this [section](#test-the-running-application).

### Quick Start Using HoloHub Container

This is the simplest and fastest way to start the application in a HoloHub dev container and get it ready to listen to request messages.

**_Note_**:
Please use your own FHIR server endpoint, as well as the OAuth2.0 authorization endpoint and client credential as needed.

```bash
./dev_container build_and_run fhir --run_args "--fhir_url <f_url> --auth_url <a_url> --uid <id> --secret <token>"
```

Add the additional command line option, `--container_args "-u root"`, to avoid seeing the following error (though no impact on execution)

```bash
Error processing line 1 of /usr/local/lib/python3.10/dist-packages/holoscan-2.4.0.pth:

  Traceback (most recent call last):
    File "/usr/lib/python3.10/site.py", line 192, in addpackage
      exec(line)
    File "<string>", line 1, in <module>
    File "/workspace/holohub/.local/lib/python3.10/site-packages/wheel_axle/runtime/__init__.py", line 80, in finalize
      with FileLock(lock_path):
    File "/workspace/holohub/.local/lib/python3.10/site-packages/filelock/_api.py", line 376, in __enter__
      self.acquire()
    File "/workspace/holohub/.local/lib/python3.10/site-packages/filelock/_api.py", line 332, in acquire
      self._acquire()
    File "/workspace/holohub/.local/lib/python3.10/site-packages/filelock/_unix.py", line 42, in _acquire
      fd = os.open(self.lock_file, open_flags, self._context.mode)
  PermissionError: [Errno 13] Permission denied: '/usr/local/lib/python3.10/dist-packages/holoscan-2.4.0.dist-info/axle.lck'
```

### Run the Application in Dev Container

This is a step wise way to run the application in a dev container.
```bash
./dev_container build --docker_file applications/ehr_query_llm/fhir/Dockerfile --img holoscan:fhir --verbose --no-cache
```

Optionally check the newly built image
```bash
$ docker images
REPOSITORY         TAG               IMAGE ID       CREATED          SIZE
holoscan           fhir              508140b8d446   3 minutes ago    14.1GB
```

Launch the container
```bash
./dev_container launch --img holoscan:fhir --as_root
```

Now in the container, build and run the application

```bash
root:~# pwd
/workspace/holohub

root:~# ./run clear_cache
root:~# ./run build fhir
root:~# ./run launch fhir --extra_args "--fhir_url <f_url> --auth_url <a_url> --uid <id> --secret <token>"
```

Once done, `exit` the container.

### Run the Application in the Host Dev Environment with dev_container script

First create and activate a Python virtual environment, followed with installing the dependencies

```bash
python3 -m venv .testenv
source .testenv/bin/activate
pip install -r applications/ehr_query_llm/fhir/requirements.txt
```

Build and install the application with `dev_container`
```bash
./dev_container build_and_install fhir
```

Now, run the application which is _installed_ in the `install` folder, with server URLs and credential of your own
```bash
python install/bin/fhir/python/ --fhir_url <f_url> --auth_url <a_url> --uid <id> --secret <token>
```

### Test the Running Application

Once the FHIR application has been started with one of the ways, a test application can be used to request and receive FHIR resources, namely `applications/ehr_query_llm/fhir/test_fhir_client.py`.

The test application contains hard coded patient name, patient FHIR resource ID, etc., corresponding to a specific test dataset, though can be easily modified for another dataset.

It is strongly recommended to run this test application in a Python virtual environment, which can be the same as in running the FHIR application. The following describes running it in its own environment.

```bash
echo "Assuming venv already created with `python3 -m venv .testenv`"
source .testenv/bin/activate
pip install -r applications/ehr_query_llm/domain_specific/fhir/requirements.txt
export PYTHONPATH=${PWD}
python applications/ehr_query_llm/fhir/test_fhir_client.py
```

From the menu, pick one of the choices for the resources of interest.

## Packaging the Application for Distribution and Deployment

With Holoscan CLI, an applications built with Holoscan SDK can be packaged into a Holoscan Application Package (HAP), which is an [Open Container Initiative](https://opencontainers.org/) compliant image. An HAP is well suited to be distributed for deployment on hosting platforms, be it Docker Compose, Kubernetes, or else. Please refer to [Packaging Holoscan Applications](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_packager.html) in the User Guide for more information.

This example application provides all the necessary contents for HAP packaging. It is required to perform the packaging in a Python virtual environment, with the application's dependencies installed, before running the following script to reveal specific packaging commands.
```bash
applications/ehr_query_llm/fhir/packageHAP.sh
```

Once the HAP is created, it can then be saved and restored on the target deployment host, and run with `docker run` command, shown below with to be substituted user specific parameters.
```bash
docker run -it --rm --net host holohub-fhir-x64-workstation-dgpu-linux-amd64:1.0 \
--fhir_url <f_url> \
--auth_url <a_url> \
--uid <id> \
--secret <token>
```
