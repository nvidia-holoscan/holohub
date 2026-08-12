#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit a synthetic failed test when a failure prevents CTest from running."""

import argparse
import platform
import re
import socket
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

LOG_MAX_BYTES = 256 * 1024
NIGHTLY_START_HOUR_UTC = 6


def compute_build_stamp(model: str, now: datetime | None = None) -> str:
    """Return the BuildStamp CTest uses for Experimental or Nightly models."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    current = current.astimezone(timezone.utc)

    if model == "Experimental":
        return f"{current:%Y%m%d-%H%M}-Experimental"
    if model != "Nightly":
        raise ValueError(f"Unsupported dashboard model: {model}")

    nightly_start = current.replace(
        hour=NIGHTLY_START_HOUR_UTC,
        minute=0,
        second=0,
        microsecond=0,
    )
    if current < nightly_start:
        current -= timedelta(days=1)
    return f"{current:%Y%m%d}-{NIGHTLY_START_HOUR_UTC:02d}00-Nightly"


def sanitize_for_xml(text: str) -> str:
    """Remove ANSI escapes and characters forbidden by XML 1.0."""
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)


def truncate_log(content: str, max_bytes: int = LOG_MAX_BYTES) -> str:
    """Keep the most useful tail of a log within CDash's upload limit."""
    encoded = content.encode("utf-8")
    if len(encoded) <= max_bytes:
        return content

    notice = "\n\n... [log truncated to fit CDash] ...\n\n"
    tail_budget = max_bytes - len(notice.encode("utf-8"))
    tail = encoded[-tail_budget:].decode("utf-8", errors="replace")
    return notice + tail


def read_log(path: Path) -> str:
    """Read and sanitize a build log."""
    try:
        return truncate_log(sanitize_for_xml(path.read_text(errors="replace")))
    except OSError as error:
        return f"Unable to read build log {path}: {error}"


def build_test_xml(
    *,
    build_name: str,
    build_stamp: str,
    site: str,
    test_name: str,
    command: str,
    exit_code: int,
    log: str,
    now: datetime | None = None,
) -> bytes:
    """Build a minimal CDash Test.xml containing one failed test."""
    current = now or datetime.now().astimezone()
    epoch = str(int(current.timestamp()))
    date_text = current.strftime("%b %d %H:%M %Z")

    root = ET.Element(
        "Site",
        {
            "BuildName": build_name,
            "BuildStamp": build_stamp,
            "Name": site,
            "Generator": "report_cdash_test_failure.py",
            "Hostname": socket.gethostname(),
            "OSName": platform.system(),
            "OSRelease": platform.release(),
            "OSPlatform": platform.machine(),
        },
    )
    testing = ET.SubElement(root, "Testing")
    ET.SubElement(testing, "StartDateTime").text = date_text
    ET.SubElement(testing, "StartTestTime").text = epoch
    test_list = ET.SubElement(testing, "TestList")
    ET.SubElement(test_list, "Test").text = f"./{test_name}"

    test = ET.SubElement(testing, "Test", {"Status": "failed"})
    ET.SubElement(test, "Name").text = test_name
    ET.SubElement(test, "Path").text = "."
    ET.SubElement(test, "FullName").text = f"./{test_name}"
    ET.SubElement(test, "FullCommandLine").text = command
    ET.SubElement(test, "StartTestTime").text = epoch
    results = ET.SubElement(test, "Results")

    measurements = [
        ("numeric/double", "Execution Time", "0"),
        ("numeric/integer", "Exit Code", str(exit_code)),
        ("text/string", "Completion Status", "Failed"),
        ("text/string", "Command Line", command),
    ]
    for measurement_type, name, value in measurements:
        measurement = ET.SubElement(
            results,
            "NamedMeasurement",
            {"type": measurement_type, "name": name},
        )
        ET.SubElement(measurement, "Value").text = value

    measurement = ET.SubElement(results, "Measurement")
    ET.SubElement(measurement, "Value").text = log
    ET.SubElement(testing, "EndDateTime").text = date_text
    ET.SubElement(testing, "EndTestTime").text = epoch
    ET.SubElement(testing, "ElapsedMinutes").text = "0"

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def submit_xml(cdash_url: str, xml: bytes) -> bool:
    """Submit Test.xml to CDash."""
    separator = "&" if "?" in cdash_url else "?"
    request = Request(
        f"{cdash_url}{separator}FileName=Test.xml",
        data=xml,
        method="PUT",
        headers={"Content-Type": "text/xml"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            if 200 <= response.status < 300:
                print(
                    f"Submitted synthetic failed test to CDash (HTTP {response.status})"
                )
                return True
            print(f"CDash returned HTTP {response.status}", file=sys.stderr)
    except URLError as error:
        print(f"CDash submission failed: {error}", file=sys.stderr)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cdash-url", required=True)
    parser.add_argument("--build-name", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument(
        "--dashboard-model", choices=("Nightly", "Experimental"), required=True
    )
    parser.add_argument("--test-name", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    xml = build_test_xml(
        build_name=args.build_name,
        build_stamp=compute_build_stamp(args.dashboard_model),
        site=args.site,
        test_name=args.test_name,
        command=args.command,
        exit_code=args.exit_code,
        log=read_log(args.log),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(xml)

    if args.dry_run:
        print(xml.decode("utf-8"))
        return 0
    return 0 if submit_xml(args.cdash_url, xml) else 1


if __name__ == "__main__":
    sys.exit(main())
