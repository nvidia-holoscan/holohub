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
import os
import glob
import json
import sys
import jsonschema
from jsonschema import validate

def validate_json(json_data, directory):
    # Describe the schema.
    with open(directory+'/metadata.schema.json', 'r') as file:
        execute_api_schema = json.load(file)

    try:
        validate(instance=json_data, schema=execute_api_schema)
    except jsonschema.exceptions.ValidationError as err:
        return False, err

    return True, 'valid'

def validate_json_directory(directory,ignore_patterns=[]):
    # Convert json to python object.
    current_wdir = os.getcwd()

    # Check if there is a metadata.json  
    subdirs = next(os.walk(current_wdir+'/'+directory))[1]
    for subdir in subdirs:
        ignore = False
        # check if we should ignore the pattern
        for ignore_pattern in ignore_patterns:
            if ignore_pattern in subdir:
                ignore = True
        
        if ignore == False:
            count = len(glob.glob(current_wdir+'/'+directory+'/'+subdir+'/**/metadata.json', recursive=True))
            if count == 0:
                print('ERROR:'+subdir+' does not contain metadata.json file')

    # Check if the metadata is valid
    for name in glob.glob(current_wdir+'/'+directory+'/**/metadata.json', recursive=True):
        with open(name, 'r') as file:
            jsonData = json.load(file)
            is_valid, msg = validate_json(jsonData,directory)
            if is_valid:
                print(name+': valid')
            else:
                print('ERROR:'+name+': invalid')
                print(msg)        

# Validate the directories
validate_json_directory('operators')
validate_json_directory('gxf_extensions',['utils'])
validate_json_directory('applications')
