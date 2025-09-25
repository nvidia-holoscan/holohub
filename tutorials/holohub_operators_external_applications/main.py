"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # noqa: E501

from holoscan.core import Application

from holohub.aja_source import AJASourceOp


class App(Application):
    def compose(self):
        # Create an instance of the AJA source operator
        aja_source = AJASourceOp(self, name="aja")

        # Add the operator to your application
        self.add_operator(aja_source)


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    main()
