# SPDX-FileCopyrightText: Copyright (c) 2026 Real-Time Innovations, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("rti.connextdds")
pytest.importorskip("holoscan")

sys.path.insert(0, str(Path(__file__).parents[2]))

from connext_dds.publisher import ConnextDDSPublisherOp  # noqa: E402
from connext_dds.subscriber import ConnextDDSSubscriberOp  # noqa: E402


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
        op.compute(op_input, op_output, execution_context)
        writer_class.return_value.write.assert_called_once_with(sample)


def test_publisher_rejects_wrong_topic_type(fragment, op_output, execution_context):
    op = ConnextDDSPublisherOp(fragment, domain_id=7, topic="Sample", topic_type=Sample)
    op._writer = MagicMock()
    op_input = MagicMock()
    op_input.receive.return_value = object()

    with pytest.raises(TypeError, match="Expected a Sample sample"):
        op.compute(op_input, op_output, execution_context)


def test_subscriber_emits_received_sample(fragment, op_output, execution_context):
    sample = Sample()
    op = ConnextDDSSubscriberOp(fragment, domain_id=7, topic="Sample", topic_type=Sample)
    op._selector = MagicMock()
    op._selector.take_data.return_value = [sample]

    op.compute(MagicMock(), op_output, execution_context)

    assert op_output.emitted == (sample, "output")
