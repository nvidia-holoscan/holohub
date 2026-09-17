#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from unittest.mock import patch

import ci.report_cdash_test_failure as cdash_report
from ci.report_cdash_test_failure import (
    build_test_xml,
    compute_build_stamp,
    sanitize_for_xml,
    submit_xml,
    truncate_log,
)


class ReportCdashTestFailureTest(unittest.TestCase):
    def test_build_stamps_match_ctest_models(self):
        before_start = datetime(2026, 8, 1, 5, 59, tzinfo=timezone.utc)
        after_start = datetime(2026, 8, 1, 6, 1, tzinfo=timezone.utc)

        self.assertEqual(
            compute_build_stamp("Nightly", before_start), "20260731-0600-Nightly"
        )
        self.assertEqual(
            compute_build_stamp("Nightly", after_start), "20260801-0600-Nightly"
        )
        self.assertEqual(
            compute_build_stamp("Experimental", after_start),
            "20260801-0601-Experimental",
        )

    def test_xml_contains_one_failed_test_and_escaped_log(self):
        xml = build_test_xml(
            build_name="camera-build",
            build_stamp="20260801-0600-Nightly",
            site="worker-1",
            test_name="holoscan_camera_v4l2.container_build",
            command="./holoscan_camera build holoscan_camera_v4l2",
            exit_code=17,
            log=sanitize_for_xml("failed <here>\x1b[31m!\x1b[0m\x00"),
            now=datetime(2026, 8, 1, 6, 2, tzinfo=timezone.utc),
        )
        root = ET.fromstring(xml)
        test = root.find("./Testing/Test[@Status='failed']")

        self.assertEqual(root.attrib["BuildName"], "camera-build")
        self.assertEqual(root.attrib["BuildStamp"], "20260801-0600-Nightly")
        self.assertEqual(test.findtext("Name"), "holoscan_camera_v4l2.container_build")
        self.assertEqual(test.findtext("Results/Measurement/Value"), "failed <here>!")
        measurements = {
            item.attrib["name"]: item.findtext("Value")
            for item in test.findall("Results/NamedMeasurement")
        }
        self.assertEqual(measurements["Exit Code"], "17")
        self.assertEqual(measurements["Completion Status"], "Failed")

    def test_log_truncation_stays_within_limit(self):
        truncated = truncate_log("é" * 100, max_bytes=80)

        self.assertLessEqual(len(truncated.encode("utf-8")), 80)
        self.assertIn("log truncated", truncated)

    def test_submit_uses_test_xml_filename(self):
        received = {"closed": False}

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                received["closed"] = True

        def fake_urlopen(request, timeout):
            received["url"] = request.full_url
            received["method"] = request.get_method()
            received["body"] = request.data
            received["content_type"] = request.headers["Content-type"]
            received["timeout"] = timeout
            return Response()

        url = "http://127.0.0.1/submit.php?project=test"
        with patch.object(cdash_report, "urlopen", fake_urlopen):
            self.assertTrue(submit_xml(url, b"<Site />"))

        self.assertEqual(
            received["url"],
            "http://127.0.0.1/submit.php?project=test&FileName=Test.xml",
        )
        self.assertEqual(received["method"], "PUT")
        self.assertEqual(received["body"], b"<Site />")
        self.assertEqual(received["content_type"], "text/xml")
        self.assertEqual(received["timeout"], 30)
        self.assertTrue(received["closed"])


if __name__ == "__main__":
    unittest.main()
