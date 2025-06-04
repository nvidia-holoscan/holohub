# FHIR Resource Sanitizer Operator

## Description

The FHIR Resource Sanitizer Operator is a Holoscan operator that processes and sanitizes FHIR (Fast Healthcare Interoperability Resources) medical records. It transforms raw FHIR resources into a more standardized and AI-friendly format while maintaining the essential medical information.

## Features

- Sanitization of various FHIR resource types
- Standardization of medical record formats
- Support for multiple resource types including:
  - Patient
  - Observation
  - Condition
  - DiagnosticReport
  - ImagingStudy
  - And more
- Error handling and logging

## Input/Output

### Input

- `records`: A FHIRQueryResponse object containing patient medical records

### Output

- `out`: A sanitized FHIRQueryResponse object with standardized medical records

## Parameters

- `fhir_endpoint` (str): FHIR service endpoint URL (default: "<http://localhost:8080/>")

## Dependencies

- holoscan
- fhir.resources
- pydantic

## Example Usage

```python
from holoscan.core import Application, OperatorSpec
from operators.ehr_query_llm.fhir.fhir_resource_sanitizer_op import FhirResourceSanitizerOp

app = Application()
app.add_operator(FhirResourceSanitizerOp(
    fragment=app.fragment,
    fhir_endpoint="https://fhir.example.com/"
))
```

## License

Apache License 2.0
