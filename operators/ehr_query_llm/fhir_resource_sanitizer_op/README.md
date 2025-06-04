# FHIR Resource Sanitizer Operator

A Holoscan operator that processes and sanitizes FHIR medical records into a standardized, AI-friendly format while maintaining essential medical information.

## Overview

The FHIR Resource Sanitizer Operator is designed to transform raw FHIR (Fast Healthcare Interoperability Resources) medical records into a more standardized format suitable for AI processing. It handles various FHIR resource types including Patient, Observation, Condition, DiagnosticReport, ImagingStudy, and more, while implementing robust error handling and logging mechanisms.

## Requirements

- holoscan
- fhir.resources
- pydantic

## Example Usage

Please check [fhir_client.py](../../../applications/ehr_query_llm/fhir/fhir_client.py) in [Generative AI Application on Holoscan integrating with FHIR Services](../../../applications/ehr_query_llm/README.md).

## Name Input/Output

- Input:  `records`: A FHIRQueryResponse object containing patient medical records

- Output: `out`: A sanitized FHIRQueryResponse object with standardized medical records

## Parameters

- `fhir_endpoint` (str): FHIR service endpoint URL (default: "<http://localhost:8080/>")
