# ZeroMQ Subscriber Operator

A Holoscan operator that subscribes to messages from a ZeroMQ message queue using the PUB/SUB pattern.

## Overview

The ZeroMQ Subscriber Operator provides a standardized interface for receiving messages from a ZeroMQ message queue. It enables applications to receive messages from publishers while handling connection management and error handling.

## Requirements

- holoscan
- pyzmq

## Example Usage

Please check [fhir_client.py](../../../applications/ehr_query_llm/fhir/fhir_client.py) in [Generative AI Application on Holoscan integrating with FHIR Services](../../../applications/ehr_query_llm/README.md).

## Name Input/Output

- Input: None
- Output: `request`: Message received from ZeroMQ

## Parameters

- `topic` (str): Topic name for message filtering
- `queue_endpoint` (str): ZeroMQ endpoint URL (e.g., "tcp://localhost:5556")
- `blocking` (bool): Whether to use blocking receive (default: False) 