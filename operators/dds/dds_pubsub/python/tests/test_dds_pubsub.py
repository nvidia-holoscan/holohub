# SPDX-FileCopyrightText: Copyright (c) 2026 Real-Time Innovations, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("rti.connextdds")
pytest.importorskip("holoscan")

sys.path.insert(0, str(Path(__file__).parents[1]))

from dds.publisher import ConnextDDSPublisherOp
from dds.subscriber import ConnextDDSSubscriberOp


class Sample:
    pass


def test_publisher_accepts_arbitrary_topic_type(fragment, op_output, execution_context):
    sample = Sample()
    op = ConnextDDSPublisherOp(fragment, domain_id=7, topic="Sample", topic_type=Sample)
    op_input = MagicMock()
    op_input.receive.return_value = sample

    with (
        patch("rti.connextdds.DomainParticipant"),
        patch("rti.connextdds.Topic"),
        patch("rti.connextdds.DataWriter") as writer_class,
    ):
        op.start()
        writer_class.assert_called_once_with(op._participant.implicit_publisher, op._dds_topic)
        op.compute(op_input, op_output, execution_context)
        writer_class.return_value.write.assert_called_once_with(sample)


def test_publisher_rejects_wrong_topic_type(fragment, op_output, execution_context):
    op = ConnextDDSPublisherOp(fragment, domain_id=7, topic="Sample", topic_type=Sample)
    op._writer = MagicMock()
    op_input = MagicMock()
    op_input.receive.return_value = object()

    with pytest.raises(TypeError, match="Expected a Sample sample"):
        op.compute(op_input, op_output, execution_context)


def test_subscriber_uses_default_qos(fragment):
    op = ConnextDDSSubscriberOp(fragment, domain_id=7, topic="Sample", topic_type=Sample)

    with (
        patch("rti.connextdds.DomainParticipant"),
        patch("rti.connextdds.Topic"),
        patch("rti.connextdds.DataReader") as reader_class,
    ):
        op.start()
        reader_class.assert_called_once_with(op._participant.implicit_subscriber, op._dds_topic)


def test_subscriber_emits_received_sample(fragment, op_output, execution_context):
    sample = Sample()
    op = ConnextDDSSubscriberOp(fragment, domain_id=7, topic="Sample", topic_type=Sample)
    op._selector = MagicMock()
    op._selector.take_data.return_value = [sample]

    op.compute(MagicMock(), op_output, execution_context)

    assert op_output.emitted == (sample, "output")
