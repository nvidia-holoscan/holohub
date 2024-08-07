# Copyright 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
from typing import Dict, Optional


class Domain(ABC):
    """Domain Class."""

    def __init__(self, metadata: Optional[Dict] = None):
        """Initialize a Domain object.

        Args:
            metadata (Optional[Dict]): A metadata.
        """
        super().__init__()

        if metadata is not None:
            self._metadata = metadata
        else:
            self._metadata = {}

    def metadata(self) -> Dict:
        return self._metadata
