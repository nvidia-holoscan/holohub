# SPDX-FileCopyrightText: Copyright (c) 2024 UNIVERSITY OF BRITISH COLUMBIA. All rights reserved.
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

import argparse
import os
from code import DAGResponseTime, MakeVars, RunExps, simulator, visualize


def main():

    # Set up the parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iterations",
        type=int,
        help="The number of iterations to run for each experiment. Default = 10",
        default=10,
    )
    parser.add_argument("-m", "--main", action="store_true", help="Enable main experiments")
    parser.add_argument("-o", "--overhead", action="store_true", help="Enable overhead experiments")
    parser.add_argument(
        "-s", "--scalability", action="store_true", help="Enable  scalability experiments"
    )
    parser.add_argument("-b", "--build", action="store_true", help="Build executables")
    parser.add_argument(
        "--numvars",
        type=int,
        help="The number of variations to use for each graph. Default = 10",
        default=10,
    )

    args = parser.parse_args()

    numvars = args.numvars

    if args.build:

        MakeVars.main(args.iterations, False)

        MakeVars.main(args.iterations, True)

    if args.main:

        graphinfo = DAGResponseTime.main("rawgraph.txt", numvars)

        RunExps.main(graphinfo, numvars)

        simulator.runwithfile("rawgraph.txt", numvars)

        visualize.main(graphinfo)

        os.system("mv observedresponsetimes.txt data/observedresponsetimes.txt")
        os.system("mv generatedexectimes.txt data/generatedexectimes.txt")
        os.system("mv predictedresponsetimes.txt data/predictedresponsetimes.txt")
        os.system("mv simulatedresponsetimes.txt data/simulatedresponsetimes.txt")
        os.system("mv evalpessimism.pdf data/evalpessimism.pdf")
        os.system("mv evalsim.pdf data/evalsim.pdf")

    if args.overhead:

        RunExps.overheadmain()

        visualize.overheadmain()

        os.system("mv observedoverheads.txt data/observedoverheads.txt")
        os.system("mv evaloverhead.pdf data/evaloverhead.pdf")

    if args.scalability:

        graphinfo = DAGResponseTime.main("paperscalability.txt", 10, True)

        simulator.runwithfile("paperscalability.txt", 10)

        visualize.scalabilitymain(graphinfo)

        visualize.scalabilitypess()

        visualize.timinganalysis()

        os.system("mv analysistimes.txt data/analysistimes.txt")
        os.system("mv evalanalysis.pdf data/evalanalysis.pdf")
        os.system("mv generatedexectimes.txt data/generatedexectimesscale.txt")
        os.system("mv predictedresponsetimes.txt data/predictedresponsetimesscale.txt")
        os.system("mv simulatedresponsetimes.txt data/simulatedresponsetimesscale.txt")
        os.system("mv evalscalability.pdf data/evalscalability.pdf")
        os.system("mv evalscalabilitypess.pdf data/evalscalabilitypess.pdf")


if __name__ == "__main__":
    main()
