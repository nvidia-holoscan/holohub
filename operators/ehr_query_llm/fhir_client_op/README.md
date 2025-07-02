# FHIR Client Operator

A Holoscan operator that enables seamless interaction with FHIR (Fast Healthcare Interoperability Resources) services for querying and retrieving patient medical records.

## Overview

The FHIR Client Operator provides a standardized interface for healthcare data exchange through FHIR services. It enables applications to securely query patient medical records while handling authentication, resource management, and error handling. The operator is designed to work with any FHIR-compliant server and supports various FHIR resource types.

Key features:

- FHIR service querying with patient search capabilities
- OAuth2 authentication support
- Configurable FHIR endpoint
- Support for various FHIR resource types
- Comprehensive error handling and logging

## Requirements

- holoscan
- requests
- fhir.resources

## Example Usage

Please check [fhir_client.py](../../../applications/ehr_query_llm/fhir/fhir_client.py) in [Generative AI Application on Holoscan integrating with FHIR Services](../../../applications/ehr_query_llm/README.md).

### Name Input/Output

- Input: `request` - JSON representation of the FHIRQuery object containing search parameters
- Output: `out` - FHIRQueryResponse object containing the original request ID and matching patient records

### Parameters

- `fhir_endpoint` (str): FHIR service endpoint URL (default: "<http://localhost:8080/>")
- `token_provider` (TokenProvider): Optional OAuth2 token provider for authentication
- `verify_cert` (bool): Whether to verify server certificates (default: True)
