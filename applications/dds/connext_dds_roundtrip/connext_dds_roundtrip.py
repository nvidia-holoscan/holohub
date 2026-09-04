# SPDX-FileCopyrightText: Copyright (c) 2026 Real-Time Innovations, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a DDS round trip within one Holoscan application."""

import logging

from connext_dds_common import (
    Command,
    Telemetry,
    add_publishers,
    add_subscribers,
    validate_publisher_sources,
)
from holoscan.core import Application

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ConnextDDSExample")


class ConnextDDSRoundTripApp(Application):
    """Publish and receive both example streams in one process."""

    def compose(self):
        self.sources = add_publishers(self)
        self.tracker = add_subscribers(self)


if __name__ == "__main__":
    app = ConnextDDSRoundTripApp()
    app.run()
    validate_publisher_sources(app.sources)
    app.tracker.validate()
    logger.info(
        "DDS round trip succeeded: Telemetry=%d, Command=%d samples",
        app.tracker.count(Telemetry),
        app.tracker.count(Command),
    )
