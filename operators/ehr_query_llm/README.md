# EHR Query LLM Operator

## Overview

The EHR Query LLM Operators are a Holoscan operator that provides a robust interface for querying and processing Electronic Health Records (EHR) using the FHIR (Fast Healthcare Interoperability Resources) standard. It enables seamless integration with FHIR services, supports OAuth2 authentication, and provides standardized medical record processing capabilities.

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
