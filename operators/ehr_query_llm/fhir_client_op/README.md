# FHIR Client Operator

## Description
The FHIR Client Operator is a Holoscan operator that interfaces with FHIR (Fast Healthcare Interoperability Resources) services to query and retrieve patient medical records. It supports authentication via OAuth2 and can handle various FHIR resource types.

## Features
- FHIR service querying with support for patient search
- OAuth2 authentication support
- Configurable FHIR endpoint
- Support for various FHIR resource types
- Error handling and logging

## Input/Output
### Input
- `request`: A JSON representation of the FHIRQuery object containing search parameters

### Output
- `out`: A FHIRQueryResponse object containing the original request ID and matching patient records

## Parameters
- `fhir_endpoint` (str): FHIR service endpoint URL (default: "http://localhost:8080/")
- `token_provider` (TokenProvider): Optional OAuth2 token provider for authentication
- `verify_cert` (bool): Whether to verify server certificates (default: True)

## Dependencies
- holoscan
- requests
- fhir.resources

## Example Usage
```python
from holoscan.core import Application, OperatorSpec
from operators.ehr_query_llm.fhir.fhir_client_op import FhirClientOperator

app = Application()
app.add_operator(FhirClientOperator(
    fragment=app.fragment,
    fhir_endpoint="https://fhir.example.com/",
    token_provider=token_provider
))
```

## License
Apache License 2.0 