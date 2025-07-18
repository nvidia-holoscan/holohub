# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from threading import Event, Thread

import cupy as cp
from holoscan.core import Operator, OperatorSpec
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


class Florence2Operator(Operator):
    def __init__(self, fragment, model_path, *args, **kwargs):
        """
        Initialize the Florence2Operator.
        """
        self.model_path = model_path
        self.task = "Object Detection"
        self.prompt = ""
        self.task_map = {
            "Caption": "<CAPTION>",
            "Detailed Caption": "<DETAILED_CAPTION>",
            "More Detailed Caption": "<MORE_DETAILED_CAPTION>",
            "Object Detection": "<OD>",
            "Dense Region Caption": "<DENSE_REGION_CAPTION>",
            "Region Proposal": "<REGION_PROPOSAL>",
            "Caption to Phrase Grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
            "Referring Expression Segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
            "Open Vocabulary Detection": "<OPEN_VOCABULARY_DETECTION>",
            "OCR": "<OCR>",
            "OCR with Region": "<OCR_WITH_REGION>",
        }
        self.task_only = {
            "<OD>",
            "<REGION_PROPOSAL>",
            "<CAPTION>",
            "<DETAILED_CAPTION>",
            "<MORE_DETAILED_CAPTION>",
            "<DENSE_REGION_CAPTION>",
            "<OCR>",
            "<OCR_WITH_REGION>",
        }
        self.output = {"output": None, "task": None}
        self.image_tensor = None
        self.is_running = Event()
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """
        Define the operator's inputs and outputs.

        Args:
            spec: The operator specification.
        """
        spec.input("video_stream")
        spec.output("output")
        spec.output("video_frame")

    def start(self):
        """
        Initialize the model and processor from the local model path.
        """
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_path, local_files_only=True, trust_remote_code=True
            )
            .eval()
            .cuda()
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=True
        )

    def run_inference(self, task_prompt, text_input, image_tensor):
        """
        Run inference on the input image using the specified task and prompt.

        Args:
            task_prompt: The task prompt for the model.
            text_input: The additional text input for the model.
            image_tensor: The input image tensor.
        """
        cp_image = cp.from_dlpack(image_tensor.get(""))
        np_image = cp.asnumpy(cp_image)
        image = Image.fromarray(np_image)

        if text_input is not None and task_prompt not in self.task_only:
            prompt = task_prompt + text_input
        else:
            prompt = task_prompt

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )
        self.output = {"output": parsed_answer, "task": task_prompt}
        self.image_tensor = image_tensor
        self.is_running.clear()

    def compute(self, op_input, op_output, context):
        """
        Compute method to receive input image and start the inference thread.
        """
        image = op_input.receive("video_stream")
        task_prompt = self.task_map[self.task]
        text_input = self.prompt

        # Save the initial image tensor if not already set
        if self.image_tensor is None:
            self.image_tensor = image

        # Start the inference thread if not already running
        if not self.is_running.is_set():
            self.is_running.set()
            run_inference_thread = Thread(
                target=self.run_inference,
                args=(
                    task_prompt,
                    text_input,
                    image,
                ),
            )
            run_inference_thread.start()

        # Emit the output and the latest video frame
        op_output.emit(self.output, "output")
        op_output.emit(self.image_tensor, "video_frame")
