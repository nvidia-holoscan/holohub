# ZeroMQ Publisher Operator

A Holoscan operator that publishes messages to a ZeroMQ message queue using the PUB/SUB pattern.

## Overview

The ZeroMQ Publisher Operator provides a standardized interface for publishing messages to a ZeroMQ message queue. It enables applications to send messages to subscribers while handling connection management and error handling.

## Requirements

- holoscan
- pyzmq

## Example Usage

Please check [fhir_client.py](../../../applications/ehr_query_llm/fhir/fhir_client.py) in [Generative AI Application on Holoscan integrating with FHIR Services](../../../applications/ehr_query_llm/README.md).

## Name Input/Output

- Input: `message`: Message to be published to ZeroMQ
- Output: None

## Parameters

- `topic` (str): Topic name for message filtering
- `queue_endpoint` (str): ZeroMQ endpoint URL (e.g., "tcp://*:5556") 