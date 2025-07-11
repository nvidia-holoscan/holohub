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

import hashlib
from pathlib import Path
from typing import Callable, Union


def checksum(
    path: Union[str, Path], hash_fn: str = "sha256", chunk_num_blocks=8192, **kwargs
) -> str:
    """Return checksum of file or directory.

    Args:
        path (Union[str, Path]): A path to file or directory.
        hash_fn (str): A hash function to use. Defaults to 'sha256'.
        chunk_num_blocks (int): A number of blocks to read at once. Defaults to 8192.
        **kwargs: Additional arguments to pass to hash function.

    Returns:
        str: checksum of file or directory
    """

    if hasattr(hashlib, hash_fn):
        hash_func: Callable = getattr(hashlib, hash_fn)
    else:
        raise ValueError("Unknown hash function")

    hashlib.blake2b
    h: hashlib._Hash = hash_func(**kwargs)
    path = Path(path)

    if path.is_file():
        path_list = [path]
    else:
        path_list = sorted(path.glob("**/*"))

    for path in path_list:
        if not path.is_file():
            continue

        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_num_blocks * h.block_size), b""):
                h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    import sys

    print(checksum(sys.argv[1]))
