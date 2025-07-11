# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import re
from typing import Union

BYTES_UNIT = {
    "kib": 2**10,
    "mib": 2**20,
    "gib": 2**30,
    "tib": 2**40,
    "pib": 2**50,
    "eib": 2**60,
    "zib": 2**70,
    "yib": 2**80,
    "ki": 2**10,
    "mi": 2**20,
    "gi": 2**30,
    "ti": 2**40,
    "pi": 2**50,
    "ei": 2**60,
    "zi": 2**70,
    "yi": 2**80,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "tb": 10**12,
    "pb": 10**15,
    "eb": 10**18,
    "zb": 10**21,
    "yb": 10**24,
    "b": 1,
}


def get_bytes(size: Union[str, int]) -> int:
    """Converts string representation of bytes to a number of bytes.

    If an integer is passed, it is returned as is (no conversion).

    Args:
        size (Union[str, int]): A string or integer representation of bytes to be converted.
                                (eg. "0.3 Gib", "3mb", "1024", 65536)

    Returns:
        int: A number of bytes represented by the input string or integer.

    Exceptions:
        ValueError: If the input string cannot be converted to an integer.
        TypeError: If the input string is not a string or integer.
    """
    if isinstance(size, int):
        if size < 0:
            raise ValueError("Negative size not allowed.")
        return size
    if not isinstance(size, str):
        raise TypeError("Size must be a string or integer.")

    m = re.match(
        r"^\s*(?P<size>(([1-9]\d+)|\d)(\.\d+)?)\s*(?P<unit>[a-z]{1,3})?\s*$", size, re.IGNORECASE
    )
    if not m:
        raise ValueError(f"Invalid size string ({size!r}).")

    parsed_size = float(m.group("size"))

    unit_match = m.group("unit")
    if unit_match:
        parsed_unit = unit_match.lower()
    else:
        parsed_unit = "b"  # default to bytes

    if parsed_unit not in BYTES_UNIT:
        raise ValueError(f"Invalid unit ({parsed_unit!r}).")

    return int(parsed_size * BYTES_UNIT[parsed_unit])


def convert_bytes(num_bytes: int, unit: str = "Mi") -> Union[str, int]:
    """Converts a number of bytes to a string representation.

    By default, the output is in MiB('Mi') format.
    If unit is 'b', the output would be an integer value.
    (e.g., convert_bytes(1024, 'b') -> 1024)

    Only one decimal point would be rendered for the output string and the decimal point would be removed if
    the first number of decimals is zero.

    e.g., convert_bytes(int(1024 * 0.211), "kib") == "0.2kib"
          convert_bytes(int(1024 * 1024), "kib") == "1024kib"

    Args:
        num_bytes (int): A number of bytes to be converted.
        unit (str): A unit to be used for the output string.

    Returns:
        Union[str, int]: A string or integer (if unit is 'b') representation of the input
                         number of bytes with the desired unit.

    Exceptions:
        ValueError: If the input number of bytes cannot be converted to the string with the desired unit.
        TypeError: If the input number of bytes is not an integer.
    """

    if not isinstance(num_bytes, int):
        raise TypeError("Size must be an integer.")
    if num_bytes < 0:
        raise ValueError("Negative size not allowed.")

    unit_lowered = unit.lower()
    if unit_lowered not in BYTES_UNIT:
        raise ValueError(f"Invalid unit ({unit!r}).")

    converted = float(num_bytes) / BYTES_UNIT[unit_lowered] * 10

    if unit_lowered == "b":
        return int(converted) // 10  # return as integer

    if int(converted) % 10 == 0:
        return f"{int(converted) // 10}{unit}"  # no decimal point
    else:
        return f"{converted / 10:.1f}{unit}"  # with decimal point (one decimal place)
