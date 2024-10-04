# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import re
from pathlib import Path

import yaml

def clean_json_string(s):
    s = s.strip()
    s = re.sub(r'\}"?$', "}", s)
    s = re.sub(r'([\{\,]\s*)([^":\s]+)(\s*:)', r'\1"\2"\3', s)
    s = re.sub(r'(:\s*)([^",\s]+)(\s*[\,\}])', r'\1"\2"\2', s)
    return s


def merge_grammars(grammar_dir="tools"):
    grammar_names = []
    grammar_rules = []
    directory_path = Path(os.path.join(Path(__file__).parent, grammar_dir))
    for tool_file in directory_path.iterdir():
        if str(tool_file).endswith(".yaml"):
            with open(str(tool_file), "r") as f:  # Convert tool_file to string
                tool_config = yaml.safe_load(f)
                tool_grammar = tool_config["grammar"]
                grammar_names.append(tool_config["name"])
                grammar_rules += tool_grammar.split("\n")

    deduplicated_grammar = remove_duplicates(grammar_rules)
    deduplicated_grammar = [grammar for grammar in deduplicated_grammar if grammar != ""]
    grammar_str = "\n".join(deduplicated_grammar)
    root_str = '\nroot ::= "{" space ( '
    is_first = True
    for grammar_name in grammar_names:
        if not is_first:
            root_str += " | "
        else:
            is_first = False
        root_str += grammar_name
    root_str += ' ) space "}"'
    grammar_str += root_str
    return grammar_str


def remove_duplicates(strings):
    seen = set()
    result = []

    for string in strings:
        if string not in seen:
            seen.add(string)
            result.append(string)

    return list(result)
