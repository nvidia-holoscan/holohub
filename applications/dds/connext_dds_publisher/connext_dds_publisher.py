# SPDX-FileCopyrightText: Copyright (c) 2026 Real-Time Innovations, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish two application-defined DDS types from a Holoscan application."""

import logging

from connext_dds_common import add_publishers, validate_publisher_sources
from holoscan.core import Application

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ConnextDDSExample")


class ConnextDDSPublisherApp(Application):
    """Publish the Telemetry and Command example streams."""

    def compose(self):
        self.sources = add_publishers(self)


if __name__ == "__main__":
    app = ConnextDDSPublisherApp()
    app.run()
    validate_publisher_sources(app.sources)
    logger.info("DDS publisher succeeded: Telemetry=20, Command=20 samples")
