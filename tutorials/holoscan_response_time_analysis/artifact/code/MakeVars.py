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

iter = 100


def main(COUNT, overhead):

    # Define the base file path and the variations
    if overhead:
        variations = defineovervar(COUNT)
    else:
        variations = definevar(COUNT)

    file_path = "base.cpp"

    # Read the original file content
    with open(file_path, "r") as file:
        content = file.readlines()

    # Loop through the variations
    for i, spec in enumerate(variations):
        # Modify the specific line (assuming it's line 2 for this example)
        modified_content = content[:326] + [spec + "\n"] + content[327:]

        # Write the modified content to a temporary file
        temp_file_path = "experiment.cpp"
        with open(temp_file_path, "w") as temp_file:
            temp_file.writelines(modified_content)

        # Compile the temporary file
        os.system("cmake --build build -j")

        if overhead:
            os.system(f"mv build/run_exp build/overheadgraph{i+1}")
        else:
            os.system(f"mv build/run_exp build/graph{i+1}")

        # Clean up the temporary file if needed
        os.remove(temp_file_path)


def definevar(COUNT):
    variations = [
        f"""// Define the operators
auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>({COUNT}));
auto replayer = make_operator<ops::PingMxTwoOutputDownstreamOp>("replayer", from_config("replayer"));
auto viz_preprocessor = make_operator<ops::PingMxDownstreamOp>("viz_preprocessor", from_config("viz_preprocessor"));
auto clahe = make_operator<ops::PingMxDownstreamOp>("clahe", from_config("clahe"));
auto preprocessor = make_operator<ops::PingMxDownstreamOp>("preprocessor", from_config("preprocessor"));
auto inference = make_operator<ops::PingMxDownstreamOp>("inference", from_config("inference"));
auto postprocessor = make_operator<ops::PingMultiInputMxOp>("postprocessor", from_config("postprocessor"));
auto holoviz = make_operator<ops::PingRxOp>("holoviz", from_config("visualizer"));

// Define the workflow
add_flow(tx, replayer, {{{{"out", "in"}}}});
add_flow(replayer, clahe, {{{{"out1", "in"}}}});
add_flow(replayer, viz_preprocessor, {{{{"out2", "in"}}}});
add_flow(clahe, preprocessor, {{{{"out", "in"}}}});
add_flow(viz_preprocessor, postprocessor, {{{{"out", "receivers"}}}});
add_flow(preprocessor, inference, {{{{"out", "in"}}}});
add_flow(inference, postprocessor, {{{{"out", "receivers"}}}});
add_flow(postprocessor, holoviz, {{{{"out", "receivers"}}}});""",
        f"""// Define the operators
auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>({COUNT}));
auto replayer = make_operator<ops::PingMxDownstreamOp>("replayer", from_config("replayer"));
auto preprocessor = make_operator<ops::PingMxTwoOutputDownstreamOp>("preprocessor", from_config("preprocessor"));
auto inference = make_operator<ops::PingMxDownstreamOp>("inference", from_config("inference"));
auto postprocessor = make_operator<ops::PingMultiInputMxOp>("postprocessor", from_config("postprocessor"));
auto holoviz = make_operator<ops::PingRxOp>("holoviz", from_config("visualizer"));

// Define the workflow
add_flow(tx, replayer, {{{{"out", "in"}}}});
add_flow(replayer, preprocessor, {{{{"out", "in"}}}});
add_flow(preprocessor, postprocessor, {{{{"out1", "receivers"}}}});
add_flow(inference, postprocessor, {{{{"out", "receivers"}}}});
add_flow(preprocessor, inference, {{{{"out2", "in"}}}});
add_flow(postprocessor, holoviz, {{{{"out", "receivers"}}}});""",
        f"""// Define the operators
auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>({COUNT}));
auto source = make_operator<ops::PingMxTwoOutputDownstreamOp>("source", from_config("source"));
auto preprocessor = make_operator<ops::PingMxDownstreamOp>("preprocessor", from_config("preprocessor"));
auto format_input = make_operator<ops::PingMxDownstreamOp>("format_input", from_config("format_input"));
auto inference = make_operator<ops::PingMxDownstreamOp>("inference", from_config("inference"));
auto postprocessor = make_operator<ops::PingMxDownstreamOp>("postprocessor", from_config("postprocessor"));
auto holoviz = make_operator<ops::PingRxOp>("holoviz", from_config("holoviz"));

// Define the workflow
add_flow(tx, source, {{{{"out", "in"}}}});
add_flow(source, preprocessor, {{{{"out1", "in"}}}});
add_flow(source, holoviz, {{{{"out2", "receivers"}}}});
add_flow(preprocessor, format_input, {{{{"out", "in"}}}});
add_flow(format_input, inference, {{{{"out", "in"}}}});
add_flow(inference, postprocessor, {{{{"out", "in"}}}});
add_flow(postprocessor, holoviz, {{{{"out", "receivers"}}}});""",
        f"""// Define the operators
auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>({COUNT}));
auto source = make_operator<ops::PingMxTwoOutputDownstreamOp>("source", from_config("source"));
auto segmentation_preprocessor = make_operator<ops::PingMxDownstreamOp>("segmentation_preprocessor", from_config("segmentation_preprocessor"));
auto segmentation_inference = make_operator<ops::PingMxDownstreamOp>("segmentation_inference", from_config("segmentation_inference"));
auto segmentation_postprocessor = make_operator<ops::PingMxDownstreamOp>("segmentation_postprocessor", from_config("segmentation_postprocessor"));
auto segmentation_visualizer = make_operator<ops::PingRxOp>("segmentation_visualizer", from_config("segmentation_visualizer"));

// Define the workflow
add_flow(tx, source, {{{{"out", "in"}}}});
add_flow(source, segmentation_preprocessor, {{{{"out1", "in"}}}});
add_flow(source, segmentation_visualizer, {{{{"out2", "receivers"}}}});
add_flow(segmentation_preprocessor, segmentation_inference, {{{{"out", "in"}}}});
add_flow(segmentation_inference, segmentation_postprocessor, {{{{"out", "in"}}}});
add_flow(segmentation_postprocessor, segmentation_visualizer, {{{{"out", "receivers"}}}});""",
        f"""// Define the operators
auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>({COUNT}));
auto source = make_operator<ops::PingMxDownstreamOp>("source", from_config("source"));
auto out_of_body_preprocessor = make_operator<ops::PingMxDownstreamOp>("out_of_body_preprocessor", from_config("out_of_body_preprocessor"));
auto out_of_body_inference = make_operator<ops::PingMxDownstreamOp>("out_of_body_inference", from_config("out_of_body_inference"));
auto out_of_body_postprocessor = make_operator<ops::PingRxOp>("out_of_body_postprocessor", from_config("out_of_body_postprocessor"));

// Define the workflow
add_flow(tx, source, {{{{"out", "in"}}}});
add_flow(source, out_of_body_preprocessor, {{{{"out", "in"}}}});
add_flow(out_of_body_preprocessor, out_of_body_inference, {{{{"out", "in"}}}});
add_flow(out_of_body_inference, out_of_body_postprocessor, {{{{"out", "receivers"}}}});""",
        f"""// Define the operators
auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>({COUNT}));
auto source = make_operator<ops::PingMxThreeOutputDownstreamOp>("source", from_config("source"));
auto format_converter = make_operator<ops::PingMxDownstreamOp>("format_converter", from_config("format_converter"));
auto format_converter_anonymization = make_operator<ops::PingMxDownstreamOp>("format_converter_anonymization", from_config("format_converter_anonymization"));
auto anonymization_preprocessor = make_operator<ops::PingMxDownstreamOp>("anonymization_preprocessor", from_config("anonymization_preprocessor"));
auto segmentation_preprocessor = make_operator<ops::PingMxDownstreamOp>("segmentation_preprocessor", from_config("segmentation_preprocessor"));
auto multi_ai_inference = make_operator<ops::PingMultiInputMxOp>("multi_ai_inference", from_config("multi_ai_inference"));
auto segmentation_postprocessor = make_operator<ops::PingMxDownstreamOp>("segmentation_postprocessor", from_config("segmentation_postprocessor"));
auto visualizer = make_operator<ops::PingRxOp>("visualizer", from_config("orsi_visualizer"));

// Define the workflow
add_flow(tx, source, {{{{"out", "in"}}}});
add_flow(source, format_converter, {{{{"out1", "in"}}}});
add_flow(source, visualizer, {{{{"out2", "receivers"}}}});
add_flow(source, format_converter_anonymization, {{{{"out3", "in"}}}});
add_flow(format_converter, segmentation_preprocessor, {{{{"out", "in"}}}});
add_flow(format_converter_anonymization, anonymization_preprocessor, {{{{"out", "in"}}}});
add_flow(anonymization_preprocessor, multi_ai_inference, {{{{"out", "receivers"}}}});
add_flow(segmentation_preprocessor, multi_ai_inference, {{{{"out", "receivers"}}}});
add_flow(multi_ai_inference, segmentation_postprocessor, {{{{"out", "in"}}}});
add_flow(multi_ai_inference, visualizer, {{{{"out", "receivers"}}}});
add_flow(segmentation_postprocessor, visualizer, {{{{"out", "receivers"}}}});""",
        f"""// Define the operators
auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>({COUNT}));
auto source = make_operator<ops::PingMxThreeOutputDownstreamOp>("source", from_config("source"));
auto detection_preprocessor = make_operator<ops::PingMxDownstreamOp>("detection_preprocessor", from_config("detection_preprocessor"));
auto segmentation_preprocessor = make_operator<ops::PingMxDownstreamOp>("segmentation_preprocessor", from_config("segmentation_preprocessor"));
auto inference = make_operator<ops::PingMultiInputOutputMxOp>("inference", from_config("inference"));
auto segmentation_postprocessor = make_operator<ops::PingMxDownstreamOp>("segmentation_postprocessor", from_config("segmentation_postprocessor"));
auto detection_postprocessor = make_operator<ops::PingMxDownstreamOp>("detection_postprocessor", from_config("detection_postprocessor"));
auto visualizer = make_operator<ops::PingRxOp>("visualizer", from_config("holoviz"));

// Define the workflow
add_flow(tx, source, {{{{"out", "in"}}}});
add_flow(source, detection_preprocessor, {{{{"out1", "in"}}}});
add_flow(source, visualizer, {{{{"out2", "receivers"}}}});
add_flow(source, segmentation_preprocessor, {{{{"out3", "in"}}}});
add_flow(detection_preprocessor, inference, {{{{"out", "receivers"}}}});
add_flow(segmentation_preprocessor, inference, {{{{"out", "receivers"}}}});
add_flow(inference, segmentation_postprocessor, {{{{"out1", "in"}}}});
add_flow(inference, detection_postprocessor, {{{{"out2", "in"}}}});
add_flow(detection_postprocessor, visualizer, {{{{"out", "receivers"}}}});
add_flow(segmentation_postprocessor, visualizer, {{{{"out", "receivers"}}}});""",
        f"""// Define the operators
auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>({COUNT}));
auto source = make_operator<ops::PingMxFourOutputDownstreamOp>("source", from_config("source"));
auto plax_cham_resized = make_operator<ops::PingMxDownstreamOp>("plax_cham_resized", from_config("plax_cham_resized"));
auto plax_cham_pre = make_operator<ops::PingMxDownstreamOp>("plax_cham_pre", from_config("plax_cham_pre"));
auto aortic_ste_pre = make_operator<ops::PingMxDownstreamOp>("aortic_ste_pre", from_config("aortic_ste_pre"));
auto b_mode_pers_pre = make_operator<ops::PingMxDownstreamOp>("b_mode_pers_pre", from_config("b_mode_pers_pre"));
auto multiai_inference = make_operator<ops::PingMultiInputMxOp>("multiai_inference", from_config("multiai_inference"));
auto multiai_postprocessor = make_operator<ops::PingMxDownstreamOp>("multiai_postprocessor", from_config("multiai_postprocessor"));
auto visualizer_icardio = make_operator<ops::PingMxDownstreamOp>("visualizer_icardio", from_config("visualizer_icardio"));
auto holoviz = make_operator<ops::PingRxOp>("holoviz", from_config("holoviz"));

// Define the workflow
add_flow(tx, source, {{{{"out", "in"}}}});
add_flow(source, plax_cham_resized, {{{{"out1", "in"}}}});
add_flow(source, plax_cham_pre, {{{{"out2", "in"}}}});
add_flow(source, aortic_ste_pre, {{{{"out3", "in"}}}});
add_flow(source, b_mode_pers_pre, {{{{"out4", "in"}}}});
add_flow(b_mode_pers_pre, multiai_inference, {{{{"out", "receivers"}}}});
add_flow(aortic_ste_pre, multiai_inference, {{{{"out", "receivers"}}}});
add_flow(plax_cham_pre, multiai_inference, {{{{"out", "receivers"}}}});
add_flow(multiai_inference, multiai_postprocessor, {{{{"out", "in"}}}});
add_flow(multiai_postprocessor, visualizer_icardio, {{{{"out", "in"}}}});
add_flow(visualizer_icardio, holoviz, {{{{"out", "receivers"}}}});
add_flow(plax_cham_resized, holoviz, {{{{"out", "receivers"}}}});""",
    ]

    return variations


def defineovervar(COUNT):

    variations = [
        f"""auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>({COUNT}));
auto rx = make_operator<ops::PingRxOp>("rx", Arg("WCET", 100));

// Define the workflow
add_flow(tx, rx, {{{{"out", "receivers"}}}});""",
        f"""auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>({COUNT}));
auto mx1 = make_operator<ops::PingMxDownstreamOp>("mx1", Arg("WCET", 100)); //, make_condition<CountCondition>(10));
auto rx = make_operator<ops::PingRxOp>("rx", Arg("WCET", 100));

// Define the workflow:  tx -> mx -> rx
add_flow(tx, mx1);
add_flow(mx1, rx, {{{{"out", "receivers"}}}});""",
        f"""auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>({COUNT}));
auto mx1 = make_operator<ops::PingMxDownstreamOp>("mx1", Arg("WCET", 100)); //, make_condition<CountCondition>(10));
auto mx2 = make_operator<ops::PingMxDownstreamOp>("mx2", Arg("WCET", 100));
auto rx = make_operator<ops::PingRxOp>("rx", Arg("WCET", 100));

// Define the workflow:  tx -> mx -> rx
add_flow(tx, mx1);
add_flow(mx1, mx2);
add_flow(mx2, rx, {{{{"out", "receivers"}}}});""",
        f"""auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>({COUNT}));
auto mx1 = make_operator<ops::PingMxDownstreamOp>("mx1", Arg("WCET", 100)); //, make_condition<CountCondition>(10));
auto mx2 = make_operator<ops::PingMxDownstreamOp>("mx2", Arg("WCET", 100));
auto mx3 = make_operator<ops::PingMxDownstreamOp>("mx3", Arg("WCET", 100));
auto rx = make_operator<ops::PingRxOp>("rx", Arg("WCET", 100));

// Define the workflow:  tx -> mx -> rx
add_flow(tx, mx1);
add_flow(mx1, mx2);
add_flow(mx2, mx3);
add_flow(mx3, rx, {{{{"out", "receivers"}}}});""",
        f"""auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>({COUNT}));
auto mx1 = make_operator<ops::PingMxDownstreamOp>("mx1", Arg("WCET", 100)); //, make_condition<CountCondition>(10));
auto mx2 = make_operator<ops::PingMxDownstreamOp>("mx2", Arg("WCET", 100));
auto mx3 = make_operator<ops::PingMxDownstreamOp>("mx3", Arg("WCET", 100));
auto mx4 = make_operator<ops::PingMxDownstreamOp>("mx4", Arg("WCET", 100));
auto rx = make_operator<ops::PingRxOp>("rx", Arg("WCET", 100));

// Define the workflow:  tx -> mx -> rx
add_flow(tx, mx1);
add_flow(mx1, mx2);
add_flow(mx2, mx3);
add_flow(mx3, mx4);
add_flow(mx4, rx, {{{{"out", "receivers"}}}});""",
        f"""auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>({COUNT}));
auto mx1 = make_operator<ops::PingMxDownstreamOp>("mx1", Arg("WCET", 100)); //, make_condition<CountCondition>(10));
auto mx2 = make_operator<ops::PingMxDownstreamOp>("mx2", Arg("WCET", 100));
auto mx3 = make_operator<ops::PingMxDownstreamOp>("mx3", Arg("WCET", 100));
auto mx4 = make_operator<ops::PingMxDownstreamOp>("mx4", Arg("WCET", 100));
auto mx5 = make_operator<ops::PingMxDownstreamOp>("mx5", Arg("WCET", 100));
auto rx = make_operator<ops::PingRxOp>("rx", Arg("WCET", 100));

// Define the workflow:  tx -> mx -> rx
add_flow(tx, mx1);
add_flow(mx1, mx2);
add_flow(mx2, mx3);
add_flow(mx3, mx4);
add_flow(mx4, mx5);
add_flow(mx5, rx, {{{{"out", "receivers"}}}});""",
        f"""auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>({COUNT}));
auto mx1 = make_operator<ops::PingMxDownstreamOp>("mx1", Arg("WCET", 100)); //, make_condition<CountCondition>(10));
auto mx2 = make_operator<ops::PingMxDownstreamOp>("mx2", Arg("WCET", 100));
auto mx3 = make_operator<ops::PingMxDownstreamOp>("mx3", Arg("WCET", 100));
auto mx4 = make_operator<ops::PingMxDownstreamOp>("mx4", Arg("WCET", 100));
auto mx5 = make_operator<ops::PingMxDownstreamOp>("mx5", Arg("WCET", 100));
auto mx6 = make_operator<ops::PingMxDownstreamOp>("mx6", Arg("WCET", 100));
auto rx = make_operator<ops::PingRxOp>("rx", Arg("WCET", 100));

// Define the workflow:  tx -> mx -> rx
add_flow(tx, mx1);
add_flow(mx1, mx2);
add_flow(mx2, mx3);
add_flow(mx3, mx4);
add_flow(mx4, mx5);
add_flow(mx5, mx6);
add_flow(mx6, rx, {{{{"out", "receivers"}}}});""",
        f"""auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>({COUNT}));
auto mx1 = make_operator<ops::PingMxDownstreamOp>("mx1", Arg("WCET", 100)); //, make_condition<CountCondition>(10));
auto mx2 = make_operator<ops::PingMxDownstreamOp>("mx2", Arg("WCET", 100));
auto mx3 = make_operator<ops::PingMxDownstreamOp>("mx3", Arg("WCET", 100));
auto mx4 = make_operator<ops::PingMxDownstreamOp>("mx4", Arg("WCET", 100));
auto mx5 = make_operator<ops::PingMxDownstreamOp>("mx5", Arg("WCET", 100));
auto mx6 = make_operator<ops::PingMxDownstreamOp>("mx6", Arg("WCET", 100));
auto mx7 = make_operator<ops::PingMxDownstreamOp>("mx7", Arg("WCET", 100));
auto rx = make_operator<ops::PingRxOp>("rx", Arg("WCET", 100));

// Define the workflow:  tx -> mx -> rx
add_flow(tx, mx1);
add_flow(mx1, mx2);
add_flow(mx2, mx3);
add_flow(mx3, mx4);
add_flow(mx4, mx5);
add_flow(mx5, mx6);
add_flow(mx6, mx7);
add_flow(mx7, rx, {{{{"out", "receivers"}}}});""",
        f"""auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>({COUNT}));
auto mx1 = make_operator<ops::PingMxDownstreamOp>("mx1", Arg("WCET", 100)); //, make_condition<CountCondition>(10));
auto mx2 = make_operator<ops::PingMxDownstreamOp>("mx2", Arg("WCET", 100));
auto mx3 = make_operator<ops::PingMxDownstreamOp>("mx3", Arg("WCET", 100));
auto mx4 = make_operator<ops::PingMxDownstreamOp>("mx4", Arg("WCET", 100));
auto mx5 = make_operator<ops::PingMxDownstreamOp>("mx5", Arg("WCET", 100));
auto mx6 = make_operator<ops::PingMxDownstreamOp>("mx6", Arg("WCET", 100));
auto mx7 = make_operator<ops::PingMxDownstreamOp>("mx7", Arg("WCET", 100));
auto mx8 = make_operator<ops::PingMxDownstreamOp>("mx8", Arg("WCET", 100));
auto rx = make_operator<ops::PingRxOp>("rx", Arg("WCET", 100));

// Define the workflow:  tx -> mx -> rx
add_flow(tx, mx1);
add_flow(mx1, mx2);
add_flow(mx2, mx3);
add_flow(mx3, mx4);
add_flow(mx4, mx5);
add_flow(mx5, mx6);
add_flow(mx6, mx7);
add_flow(mx7, mx8);
add_flow(mx8, rx, {{{{"out", "receivers"}}}});""",
        f"""auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>({COUNT}));
auto mx1 = make_operator<ops::PingMxDownstreamOp>("mx1", Arg("WCET", 100)); //, make_condition<CountCondition>(10));
auto mx2 = make_operator<ops::PingMxDownstreamOp>("mx2", Arg("WCET", 100));
auto mx3 = make_operator<ops::PingMxDownstreamOp>("mx3", Arg("WCET", 100));
auto mx4 = make_operator<ops::PingMxDownstreamOp>("mx4", Arg("WCET", 100));
auto mx5 = make_operator<ops::PingMxDownstreamOp>("mx5", Arg("WCET", 100));
auto mx6 = make_operator<ops::PingMxDownstreamOp>("mx6", Arg("WCET", 100));
auto mx7 = make_operator<ops::PingMxDownstreamOp>("mx7", Arg("WCET", 100));
auto mx8 = make_operator<ops::PingMxDownstreamOp>("mx8", Arg("WCET", 100));
auto mx9 = make_operator<ops::PingMxDownstreamOp>("mx9", Arg("WCET", 100));
auto rx = make_operator<ops::PingRxOp>("rx", Arg("WCET", 100));

// Define the workflow:  tx -> mx -> rx
add_flow(tx, mx1);
add_flow(mx1, mx2);
add_flow(mx2, mx3);
add_flow(mx3, mx4);
add_flow(mx4, mx5);
add_flow(mx5, mx6);
add_flow(mx6, mx7);
add_flow(mx7, mx8);
add_flow(mx8, mx9);
add_flow(mx9, rx, {{{{"out", "receivers"}}}});""",
        f"""auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>({COUNT}));
auto mx1 = make_operator<ops::PingMxDownstreamOp>("mx1", Arg("WCET", 100)); //, make_condition<CountCondition>(10));
auto mx2 = make_operator<ops::PingMxDownstreamOp>("mx2", Arg("WCET", 100));
auto mx3 = make_operator<ops::PingMxDownstreamOp>("mx3", Arg("WCET", 100));
auto mx4 = make_operator<ops::PingMxDownstreamOp>("mx4", Arg("WCET", 100));
auto mx5 = make_operator<ops::PingMxDownstreamOp>("mx5", Arg("WCET", 100));
auto mx6 = make_operator<ops::PingMxDownstreamOp>("mx6", Arg("WCET", 100));
auto mx7 = make_operator<ops::PingMxDownstreamOp>("mx7", Arg("WCET", 100));
auto mx8 = make_operator<ops::PingMxDownstreamOp>("mx8", Arg("WCET", 100));
auto mx9 = make_operator<ops::PingMxDownstreamOp>("mx9", Arg("WCET", 100));
auto mx10 = make_operator<ops::PingMxDownstreamOp>("mx10", Arg("WCET", 100));
auto rx = make_operator<ops::PingRxOp>("rx", Arg("WCET", 100));

// Define the workflow:  tx -> mx -> rx
add_flow(tx, mx1);
add_flow(mx1, mx2);
add_flow(mx2, mx3);
add_flow(mx3, mx4);
add_flow(mx4, mx5);
add_flow(mx5, mx6);
add_flow(mx6, mx7);
add_flow(mx7, mx8);
add_flow(mx8, mx9);
add_flow(mx9, mx10);
add_flow(mx10, rx, {{{{"out", "receivers"}}}});""",
    ]

    return variations


if __name__ == "__main__":
    main(iter, True)
