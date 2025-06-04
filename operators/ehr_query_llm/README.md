# EHR Query LLM Operator

## Overview

The EHR Query LLM Operator is a Holoscan operator that provides a robust interface for querying and processing Electronic Health Records (EHR) using the FHIR (Fast Healthcare Interoperability Resources) standard. It enables seamless integration with FHIR services, supports OAuth2 authentication, and provides standardized medical record processing capabilities.

## Features

- FHIR service querying with support for patient search
- OAuth2 authentication support
- Configurable FHIR endpoint
- Support for various FHIR resource types including:
  - Patient
  - Observation
  - Condition
  - DiagnosticReport
  - ImagingStudy
  - DocumentReference
  - And more
- ZeroMQ-based message handling for distributed systems
- Standardized medical record sanitization and processing
- Comprehensive error handling and logging

## Components

### FHIR Client Operator

- Handles FHIR service queries
- Supports patient search and resource retrieval
- Configurable authentication via OAuth2
- Processes FHIR responses into standardized format

### FHIR Resource Sanitizer Operator

- Sanitizes and standardizes FHIR resources
- Transforms raw FHIR data into AI-friendly format
- Maintains essential medical information
- Supports multiple resource types

### ZeroMQ Message Handling

- Publisher/Subscriber pattern for distributed communication
- Configurable topics and endpoints
- Support for both blocking and non-blocking operations

## Dependencies

- holoscan >= 2.5.0
- fhir.resources >= 7.0.0
- pyzmq >= 25.1.0
- requests >= 2.31.0
- pydantic >= 2.0.0

## Usage

### Basic Setup

```python
from holoscan.core import Application
from operators.ehr_query_llm.fhir.fhir_client_op import FhirClientOperator
from operators.ehr_query_llm.fhir.fhir_resource_sanitizer_op import FhirResourceSanitizerOp

app = Application()

# Add FHIR Client Operator
app.add_operator(FhirClientOperator(
    fragment=app.fragment,
    fhir_endpoint="https://fhir.example.com/",
    token_provider=token_provider  # Optional
))

# Add FHIR Resource Sanitizer Operator
app.add_operator(FhirResourceSanitizerOp(
    fragment=app.fragment,
    fhir_endpoint="https://fhir.example.com/"
))
```

### Message Handling

```python
from operators.ehr_query_llm.zero_mq.publisher_op import ZeroMQPublisherOp
from operators.ehr_query_llm.zero_mq.subscriber_op import ZeroMQSubscriberOp

# Add Publisher
app.add_operator(ZeroMQPublisherOp(
    fragment=app.fragment,
    topic="ehr_requests",
    queue_endpoint="tcp://*:5556"
))

# Add Subscriber
app.add_operator(ZeroMQSubscriberOp(
    fragment=app.fragment,
    topic="ehr_requests",
    queue_endpoint="tcp://localhost:5556"
))
```

## Configuration

### FHIR Client Operator Parameters

- `fhir_endpoint`: FHIR service endpoint URL
- `token_provider`: Optional OAuth2 token provider
- `verify_cert`: Whether to verify server certificates

### FHIR Resource Sanitizer Parameters

- `fhir_endpoint`: FHIR service endpoint URL

### ZeroMQ Parameters

- `topic`: Message topic for filtering
- `queue_endpoint`: ZeroMQ endpoint URL
- `blocking`: Whether to use blocking receive (default: False)

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For support, please open an issue in the repository or contact the Holoscan SDK team.
