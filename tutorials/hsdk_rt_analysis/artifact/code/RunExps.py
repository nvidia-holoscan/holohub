# SPDX-FileCopyrightText: Copyright (c) 2025 UNIVERSITY OF BRITISH COLUMBIA. All rights reserved.
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


# Function to parse output log and find the observed WCRT
def parselog(filepath):

    source = open(filepath, "r")

    delays = []

    for line in source:
        if "->" not in line:
            pass
        else:
            parsed = line.split(",")
            start = parsed[1]
            end = parsed[-1][:-3]
            delay = (int(end) - int(start)) / 1000
            delays.append(delay)

    source.close()

    return max(delays)


def main(graphs, numvars):
    dest = open("observedresponsetimes.txt", "w")

    with open("generatedexectimes.txt", "r") as allexectimes:
        for i, data in enumerate(graphs[0]):

            if numvars == 0:
                n = data[0]
            else:
                n = numvars

            # Blank line
            allexectimes.readline()
            for var in range(n):
                # Read the original file content
                with open("experimentbase.yaml", "r") as script:
                    scriptbase = script.readlines()
                    exectimes = []
                    for line in range((2 * data[0]) + 1):
                        exectimes.append(allexectimes.readline())

                    # Add the execution times
                    modified_content = scriptbase[:31] + exectimes + scriptbase[32:]

                    # Write the modified content to a temporary file
                    temp_file_path = "build/experiment.yaml"
                    with open(temp_file_path, "w") as temp_file:
                        temp_file.writelines(modified_content)

                    # Compile the temporary file
                    os.system(f"./build/graph{i+1}")

                    print("")
                    print("Graph " + str(i + 1) + " variation " + str(var) + " complete")
                    print("")

                    dest.write(
                        "Graph "
                        + str(i + 1)
                        + " variation "
                        + str(var)
                        + " observed WCET = "
                        + str(parselog("logger.log"))
                        + "\n"
                    )

                    # Clean up the temporary file if needed
                    os.remove(temp_file_path)

    dest.close


def overheadmain():
    dest = open("observedoverheads.txt", "w")

    with open("experimentbase.yaml", "r") as script:
        scriptbase = script.readlines()

        # Create an experiment.yaml file for basic configuration
        temp_file_path = "build/experiment.yaml"
        with open(temp_file_path, "w") as temp_file:
            temp_file.writelines(scriptbase)

    for i in range(11):

        # Run the experiment
        os.system(f"./build/overheadgraph{i+1}")

        print("")
        print("Overhead graph " + str(i + 1) + " complete")
        print("")

        dest.write(
            "Chain Length " + str(i + 1) + " observed WCRT = " + str(parselog("logger.log")) + "\n"
        )

    dest.close
