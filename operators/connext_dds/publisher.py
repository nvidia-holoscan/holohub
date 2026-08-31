# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, Real-Time Innovations, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import rti.connextdds as dds
from holoscan.core import Operator, OperatorSpec


class ConnextDDSPublisherOp(Operator):
    """Publish samples of any RTI Connext DDS-compatible Python type."""

    def __init__(self, fragment, *args, domain_id, topic, topic_type, **kwargs):
        self.domain_id = domain_id
        self.topic = topic
        self.topic_type = topic_type
        self._participant = None
        self._dds_topic = None
        self._writer = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def start(self):
        self._participant = dds.DomainParticipant(domain_id=self.domain_id)
        self._dds_topic = dds.Topic(self._participant, self.topic, self.topic_type)
        # Use Connext's default QoS, including the default XML profile if configured.
        self._writer = dds.DataWriter(self._participant.implicit_publisher, self._dds_topic)

    def compute(self, op_input, op_output, context):
        sample = op_input.receive("input")
        if not isinstance(sample, self.topic_type):
            raise TypeError(
                f"Expected a {self.topic_type.__name__} sample, got {type(sample).__name__}"
            )
        self._writer.write(sample)

    def stop(self):
        for entity_name in ("_writer", "_dds_topic", "_participant"):
            entity = getattr(self, entity_name)
            if entity is not None:
                entity.close()
                setattr(self, entity_name, None)
