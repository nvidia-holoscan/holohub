# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import glob
import json
import os
import sys

import jsonschema
from jsonschema import Draft4Validator
from referencing import Registry
from referencing.jsonschema import DRAFT4


def validate_json(json_data, directory):
    BASE_SCHEMA = "utilities/metadata/project.schema.json"

    # Describe the schema.
    with open(BASE_SCHEMA) as file:
        base_schema = json.load(file)
    registry = Registry().with_resource(base_schema["$id"], DRAFT4.create_resource(base_schema))

    with open(directory + "/metadata.schema.json", "r") as file:
        try:
            execute_api_schema = json.load(file)
        except json.decoder.JSONDecodeError as err:
            return False, err
    validator = Draft4Validator(execute_api_schema, registry=registry)

    try:
        validator.validate(json_data)
    except jsonschema.exceptions.ValidationError as err:
        return False, err

    return True, "valid"


def validate_json_directory(directory, ignore_patterns=[], metadata_is_required: bool = True):
    exit_code = 0
    # Convert json to python object.
    current_wdir = os.getcwd()

    # Check if there is a metadata.json
    subdirs = next(os.walk(current_wdir + "/" + directory))[1]
    for subdir in subdirs:
        ignore = False
        # check if we should ignore the pattern
        for ignore_pattern in ignore_patterns:
            if ignore_pattern in subdir:
                ignore = True

        if ignore is False:
            count = len(
                glob.glob(
                    current_wdir + "/" + directory + "/" + subdir + "/**/metadata.json",
                    recursive=True,
                )
            )
            if count == 0:
                if metadata_is_required:
                    print("ERROR:" + subdir + " does not contain metadata.json file")
                    exit_code = 1
                else:
                    print("WARNING:" + subdir + " does not contain metadata.json file")

    # Check if the metadata is valid
    for name in glob.glob(current_wdir + "/" + directory + "/**/metadata.json", recursive=True):
        ignore = False
        # check if we should ignore the pattern
        for ignore_pattern in ignore_patterns:
            if ignore_pattern in name:
                ignore = True
        if ignore:
            continue

        with open(name, "r") as file:
            try:
                jsonData = json.load(file)
            except json.decoder.JSONDecodeError:
                print("ERROR:" + name + ": invalid")
                exit_code = 1
                continue

            is_valid, msg = validate_json(jsonData, directory)
            if is_valid:
                print(name + ": valid")
            else:
                print("ERROR:" + name + ": invalid")
                print(msg)
                exit_code = 1

    return exit_code


# Validate the directories
if __name__ == "__main__":
    exit_code_op = validate_json_directory("operators", ignore_patterns=["template"])
    exit_code_extensions = validate_json_directory("gxf_extensions", ignore_patterns=["utils"])
    exit_code_applications = validate_json_directory("applications", ignore_patterns=["template"])
    exit_code_tutorials = validate_json_directory(
        "tutorials", ignore_patterns=["template"], metadata_is_required=False
    )
    exit_code_benchmarks = validate_json_directory("benchmarks")

    sys.exit(
        max(
            exit_code_op,
            exit_code_extensions,
            exit_code_applications,
            exit_code_tutorials,
            exit_code_benchmarks,
        )
    )
