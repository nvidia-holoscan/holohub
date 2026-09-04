# SPDX-FileCopyrightText: Copyright (c) 2026 Real-Time Innovations, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared graph components for the RTI Connext DDS example applications."""

import logging
from datetime import timedelta

from holoscan.conditions import CountCondition, PeriodicCondition
from holoscan.core import Operator, OperatorSpec
from rti import idl

from holohub.dds import ConnextDDSPublisherOp, ConnextDDSSubscriberOp

DOMAIN_ID = 42
SAMPLE_COUNT = 20
PUBLISH_PERIOD = timedelta(milliseconds=100)
SUBSCRIBE_PERIOD = timedelta(milliseconds=25)

logger = logging.getLogger("ConnextDDSExample")


@idl.struct
class Telemetry:
    """Example telemetry type owned by the application."""

    sequence: int = 0
    value: float = 0.0


@idl.struct
class Command:
    """Second application-owned type used to demonstrate generic IDL support."""

    command_id: int = 0
    enabled: bool = False


TYPE_CONFIGS = (
    (Telemetry, "HoloscanConnextTelemetry", "telemetry"),
    (Command, "HoloscanConnextCommand", "command"),
)


class SampleSourceOp(Operator):
    """Generate the finite sample sequence for either example type."""

    def __init__(self, fragment, *args, sample_type, **kwargs):
        self.sequence = 0
        self.sample_type = sample_type
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def compute(self, op_input, op_output, context):
        del op_input, context
        if self.sample_type is Telemetry:
            sample = Telemetry(sequence=self.sequence, value=self.sequence * 0.5)
        elif self.sample_type is Command:
            sample = Command(command_id=self.sequence, enabled=self.sequence % 2 == 0)
        else:
            raise TypeError(f"Unsupported example type: {self.sample_type.__name__}")

        logger.info("Publishing %s", sample)
        self.sequence += 1
        op_output.emit(sample, "output")


class SampleTracker:
    """Track and validate the samples received for both example types."""

    def __init__(self, application):
        self.application = application
        self.identifiers = {sample_type: set() for sample_type, _, _ in TYPE_CONFIGS}
        self._stopped = False

    def record(self, sample_type, sample):
        if not isinstance(sample, sample_type):
            raise TypeError(f"Expected {sample_type.__name__}, got {type(sample).__name__}")

        identifier = sample.sequence if sample_type is Telemetry else sample.command_id
        if identifier in self.identifiers[sample_type]:
            raise RuntimeError(f"Received duplicate {sample_type.__name__} identifier {identifier}")

        self.identifiers[sample_type].add(identifier)
        if self.complete and not self._stopped:
            self._stopped = True
            self.application.stop_execution()

    @property
    def complete(self):
        return all(len(values) == SAMPLE_COUNT for values in self.identifiers.values())

    def count(self, sample_type):
        return len(self.identifiers[sample_type])

    def validate(self):
        expected = set(range(SAMPLE_COUNT))
        for sample_type, identifiers in self.identifiers.items():
            if identifiers != expected:
                missing = sorted(expected - identifiers)
                unexpected = sorted(identifiers - expected)
                raise RuntimeError(
                    f"Invalid {sample_type.__name__} samples: "
                    f"missing={missing}, unexpected={unexpected}"
                )


class SampleSinkOp(Operator):
    """Record and validate samples received by a DDS subscriber operator."""

    def __init__(self, fragment, *args, sample_type, tracker, **kwargs):
        self.sample_type = sample_type
        self.tracker = tracker
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        del op_output, context
        sample = op_input.receive("input")
        logger.info("Received %s", sample)
        self.tracker.record(self.sample_type, sample)


def add_publishers(application):
    """Add publishers for both example types and return their source operators."""
    sources = {}
    for sample_type, topic, slug in TYPE_CONFIGS:
        source = SampleSourceOp(
            application,
            CountCondition(application, SAMPLE_COUNT),
            PeriodicCondition(application, recess_period=PUBLISH_PERIOD),
            sample_type=sample_type,
            name=f"{slug}_source",
        )
        publisher = ConnextDDSPublisherOp(
            application,
            domain_id=DOMAIN_ID,
            topic=topic,
            topic_type=sample_type,
            name=f"{slug}_publisher",
        )
        application.add_flow(source, publisher, {("output", "input")})
        sources[sample_type] = source
    return sources


def add_subscribers(application):
    """Add subscribers for both example types and return their shared tracker."""
    tracker = SampleTracker(application)
    for sample_type, topic, slug in TYPE_CONFIGS:
        subscriber = ConnextDDSSubscriberOp(
            application,
            PeriodicCondition(application, recess_period=SUBSCRIBE_PERIOD),
            domain_id=DOMAIN_ID,
            topic=topic,
            topic_type=sample_type,
            name=f"{slug}_subscriber",
        )
        sink = SampleSinkOp(
            application,
            sample_type=sample_type,
            tracker=tracker,
            name=f"{slug}_sink",
        )
        application.add_flow(subscriber, sink, {("output", "input")})
    return tracker


def validate_publisher_sources(sources):
    """Ensure each source emitted the configured number of samples."""
    for sample_type, source in sources.items():
        if source.sequence != SAMPLE_COUNT:
            raise RuntimeError(
                f"Expected {SAMPLE_COUNT} {sample_type.__name__} samples, "
                f"published {source.sequence}"
            )
