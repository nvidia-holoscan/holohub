# SPDX-FileCopyrightText: Copyright (c) 2026 Real-Time Innovations, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receive two application-defined DDS types in a Holoscan application."""

import logging

from connext_dds_common import Command, Telemetry, add_subscribers
from holoscan.core import Application

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ConnextDDSExample")


class ConnextDDSSubscriberApp(Application):
    """Receive and validate the Telemetry and Command example streams."""

    def compose(self):
        self.tracker = add_subscribers(self)


if __name__ == "__main__":
    app = ConnextDDSSubscriberApp()
    app.run()
    app.tracker.validate()
    logger.info(
        "DDS subscriber succeeded: Telemetry=%d, Command=%d samples",
        app.tracker.count(Telemetry),
        app.tracker.count(Command),
    )
