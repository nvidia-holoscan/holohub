# SPDX-FileCopyrightText: Copyright (c) 2026 Real-Time Innovations, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from datetime import timedelta

import rti.idl as idl
from holoscan.conditions import CountCondition, PeriodicCondition
from holoscan.core import Application, Operator, OperatorSpec

from holohub.connext_dds import ConnextDDSPublisherOp, ConnextDDSSubscriberOp


DOMAIN_ID = 42
TOPIC_NAME = "HoloscanConnextTelemetry"
COMMAND_TOPIC_NAME = "HoloscanConnextCommand"
SAMPLE_COUNT = 20

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ConnextDDSExample")


@idl.struct
class Telemetry:
    sequence: int = 0
    value: float = 0.0


@idl.struct
class Command:
    command_id: int = 0
    enabled: bool = False


class TelemetrySourceOp(Operator):
    def __init__(self, fragment, *args, sample_type=Telemetry, **kwargs):
        self.sequence = 0
        self.sample_type = sample_type
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def compute(self, op_input, op_output, context):
        if self.sample_type is Telemetry:
            sample = Telemetry(sequence=self.sequence, value=self.sequence * 0.5)
        else:
            sample = Command(command_id=self.sequence, enabled=self.sequence % 2 == 0)
        logger.info("Publishing %s", sample)
        self.sequence += 1
        op_output.emit(sample, "output")


class TelemetrySinkOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.received_count = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        sample = op_input.receive("input")
        self.received_count += 1
        logger.info("Received %s", sample)


class ConnextDDSExampleApp(Application):
    def compose(self):
        source = TelemetrySourceOp(
            self,
            CountCondition(self, SAMPLE_COUNT),
            PeriodicCondition(self, recess_period=timedelta(milliseconds=100)),
            name="telemetry_source",
        )
        publisher = ConnextDDSPublisherOp(
            self,
            domain_id=DOMAIN_ID,
            topic=TOPIC_NAME,
            topic_type=Telemetry,
            name="dds_publisher",
        )
        subscriber = ConnextDDSSubscriberOp(
            self,
            CountCondition(self, SAMPLE_COUNT * 5),
            PeriodicCondition(self, recess_period=timedelta(milliseconds=25)),
            domain_id=DOMAIN_ID,
            topic=TOPIC_NAME,
            topic_type=Telemetry,
            name="dds_subscriber",
        )
        self.sink = TelemetrySinkOp(self, name="telemetry_sink")
        command_source = TelemetrySourceOp(
            self,
            CountCondition(self, SAMPLE_COUNT),
            PeriodicCondition(self, recess_period=timedelta(milliseconds=100)),
            sample_type=Command,
            name="command_source",
        )
        command_publisher = ConnextDDSPublisherOp(
            self,
            domain_id=DOMAIN_ID,
            topic=COMMAND_TOPIC_NAME,
            topic_type=Command,
            name="command_publisher",
        )
        command_subscriber = ConnextDDSSubscriberOp(
            self,
            CountCondition(self, SAMPLE_COUNT * 5),
            PeriodicCondition(self, recess_period=timedelta(milliseconds=25)),
            domain_id=DOMAIN_ID,
            topic=COMMAND_TOPIC_NAME,
            topic_type=Command,
            name="command_subscriber",
        )
        self.command_sink = TelemetrySinkOp(self, name="command_sink")

        self.add_flow(source, publisher, {("output", "input")})
        self.add_flow(subscriber, self.sink, {("output", "input")})
        self.add_flow(command_source, command_publisher, { ("output", "input") })
        self.add_flow(command_subscriber, self.command_sink, { ("output", "input") })


if __name__ == "__main__":
    app = ConnextDDSExampleApp()
    app.run()
    if app.sink.received_count != SAMPLE_COUNT or app.command_sink.received_count != SAMPLE_COUNT:
        raise RuntimeError("Not all DDS samples were received for one or more IDL types")
    logger.info(
        "DDS round trip succeeded: Telemetry=%d, Command=%d samples",
        app.sink.received_count,
        app.command_sink.received_count,
    )
