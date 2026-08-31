# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Real-Time Innovations, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import rti.connextdds as dds
from holoscan.core import Operator, OperatorSpec


class ConnextDDSSubscriberOp(Operator):
    """Receive samples of any RTI Connext DDS-compatible Python type."""

    def __init__(self, fragment, *args, domain_id, topic, topic_type, **kwargs):
        self.domain_id = domain_id
        self.topic = topic
        self.topic_type = topic_type
        self._participant = None
        self._dds_topic = None
        self._reader = None
        self._selector = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def start(self):
        self._participant = dds.DomainParticipant(domain_id=self.domain_id)
        self._dds_topic = dds.Topic(self._participant, self.topic, self.topic_type)

        reader_qos = dds.DataReaderQos()
        reader_qos.reliability.kind = dds.ReliabilityKind.RELIABLE
        reader_qos.history.kind = dds.HistoryKind.KEEP_ALL
        reader_qos.durability.kind = dds.DurabilityKind.TRANSIENT_LOCAL

        self._reader = dds.DataReader(
            self._participant.implicit_subscriber, self._dds_topic, reader_qos
        )
        self._selector = self._reader.select().max_samples(1)

    def compute(self, op_input, op_output, context):
        for sample in self._selector.take_data():
            op_output.emit(sample, "output")

    def stop(self):
        self._selector = None
        for entity_name in ("_reader", "_dds_topic", "_participant"):
            entity = getattr(self, entity_name)
            if entity is not None:
                entity.close()
                setattr(self, entity_name, None)
