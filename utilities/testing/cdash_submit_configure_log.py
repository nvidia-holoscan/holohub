#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Report pre-CTest failures to CDash.

When a CI step fails before CTest runs (e.g. docker build, checkout), CTest
never executes and nothing appears on CDash. This utility submits a
Configure.xml (error in the Configure column) so the failure is visible on
the dashboard.

Only use this for failures — submitting to a build that CTest also submits to
will cause CDash to delete and recreate the build (RemoveIfDone behavior).

Usage:
    python3 utilities/testing/cdash_submit_configure_log.py \\
        --cdash-url "http://cdash.nvidia.com/submit.php?project=Holoscan" \\
        --build-name "holohub-linux-arm64-adv_networking_bench-dpdk-main" \\
        --site "my-node" \\
        --cmd "docker_build" \\
        --exit-code 1 \\
        --log /tmp/docker_build.log
"""

import argparse
import re
import socket
import sys
import time
import xml.dom.minidom
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from urllib.error import URLError
from urllib.request import Request, urlopen

# CDash has size limits on Configure.xml log content
CONFIGURE_LOG_MAX_BYTES = 256 * 1024  # 256 KB


def compute_build_stamp(dashboard_model):
    """Compute CDash BuildStamp matching CTest's format: YYYYMMDD-HHMM-{Model}.

    The date rolls back one day if the current UTC time is before the nightly
    start time (06:00 UTC), matching CTest's behavior.
    """
    now = datetime.now(timezone.utc)
    nightly_start = now.replace(hour=6, minute=0, second=0, microsecond=0)
    if now < nightly_start:
        now -= timedelta(days=1)
    return f"{now.strftime('%Y%m%d-%H%M')}-{dashboard_model}"


def sanitize_for_xml(text):
    """Remove characters illegal in XML 1.0 (e.g. ANSI escape codes, control chars).

    XML 1.0 only allows: #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD].
    ANSI escape sequences (ESC = 0x1B) from colored terminal output are the most
    common offender.
    """
    # Strip ANSI escape sequences first (preserve the text between them)
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)
    # Remove any remaining XML-illegal control characters (keep tab, newline, carriage return)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def read_log_file(path):
    """Read a log file, returning empty string if path is None or unreadable."""
    if not path:
        return ""
    try:
        with open(path, "r", errors="replace") as f:
            return sanitize_for_xml(f.read())
    except OSError as e:
        print(f"Warning: could not read log file {path}: {e}", file=sys.stderr)
        return ""


def truncate_log(content, max_bytes):
    """Truncate log content to max_bytes, adding a truncation notice."""
    encoded = content.encode("utf-8")
    if len(encoded) <= max_bytes:
        return content
    notice = "\n\n... [log truncated to 256KB for CDash] ...\n\n"
    # Keep the tail (most relevant for errors)
    tail_budget = max_bytes - len(notice.encode("utf-8"))
    truncated = encoded[-tail_budget:].decode("utf-8", errors="replace")
    return notice + truncated


def build_site_attribs(args, build_stamp):
    """Return common Site element attributes."""
    hostname = socket.gethostname()
    return {
        "BuildName": args.build_name,
        "BuildStamp": build_stamp,
        "Name": args.site,
        "Hostname": hostname,
        "Generator": "cdash_submit_configure_log.py",
    }


def build_configure_xml(args, build_stamp, log_content):
    """Build Configure.xml for error reporting."""
    epoch = str(int(time.time()))

    site = ET.Element("Site", build_site_attribs(args, build_stamp))
    configure = ET.SubElement(site, "Configure")

    ET.SubElement(configure, "StartConfigureTime").text = epoch
    ET.SubElement(configure, "ConfigureCommand").text = f"[before ctest could run]\n{args.cmd}"
    ET.SubElement(configure, "Log").text = truncate_log(log_content, CONFIGURE_LOG_MAX_BYTES)
    ET.SubElement(configure, "ConfigureStatus").text = str(args.exit_code)
    ET.SubElement(configure, "EndConfigureTime").text = epoch
    ET.SubElement(configure, "ElapsedMinutes").text = "0"

    return site


def xml_to_string(root):
    """Serialize an ElementTree element to an XML string with declaration."""
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=False)
    dom = xml.dom.minidom.parseString(xml_bytes)
    pretty_xml = dom.toprettyxml(indent="  ", encoding="UTF-8").decode("utf-8")
    # Remove the minidom's XML declaration and add our own
    lines = pretty_xml.split("\n")
    if lines[0].startswith("<?xml"):
        lines = lines[1:]
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + "\n".join(lines).strip() + "\n"


def submit_xml(cdash_url, filename, xml_string, dry_run=False):
    """Submit XML to CDash via HTTP PUT."""
    separator = "&" if "?" in cdash_url else "?"
    url = f"{cdash_url}{separator}FileName={filename}"

    if dry_run:
        print(f"\n=== DRY RUN: {filename} ===")
        print(f"URL: {url}")
        print(f"Content:\n{xml_string}")
        print(f"=== END {filename} ===\n")
        return True

    data = xml_string.encode("utf-8")
    request = Request(url, data=data, method="PUT")
    request.add_header("Content-Type", "text/xml")
    request.add_header("Content-Length", str(len(data)))

    try:
        with urlopen(request, timeout=30) as response:
            status = response.status
            body = response.read().decode("utf-8", errors="replace")
            if status < 200 or status >= 300:
                print(
                    f"Warning: CDash returned status {status} for {filename}: {body}",
                    file=sys.stderr,
                )
                return False
            print(f"Submitted {filename} to CDash (HTTP {status})")
            return True
    except URLError as e:
        print(f"Error submitting {filename} to CDash: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Report pre-CTest failures to CDash.")
    parser.add_argument(
        "--cdash-url",
        required=True,
        help="CDash submit URL (e.g. http://cdash.nvidia.com/submit.php?project=Holoscan)",
    )
    parser.add_argument(
        "--build-name",
        required=True,
        help="CDash build name (should match the name CTest would use)",
    )
    parser.add_argument("--site", required=True, help="CDash site name")
    parser.add_argument("--cmd", default="", help="Command associated with the log")
    parser.add_argument(
        "--exit-code", type=int, default=1, help="Exit code of failed command (default: 1)"
    )
    parser.add_argument("--log", help="Path to log file to include as the configure command output")
    parser.add_argument(
        "--dashboard-model",
        default="Nightly",
        help="CDash dashboard model: Nightly | Experimental (default: Nightly)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print XML and URL without submitting"
    )

    args = parser.parse_args()
    build_stamp = compute_build_stamp(args.dashboard_model)
    log_content = read_log_file(args.log)

    # Submit Configure.xml (error visible in Configure column)
    configure_xml = build_configure_xml(args, build_stamp, log_content)
    if not submit_xml(args.cdash_url, "Configure.xml", xml_to_string(configure_xml), args.dry_run):
        sys.exit(1)


if __name__ == "__main__":
    main()
