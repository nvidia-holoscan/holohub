# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import gradio as gr
import openai

# Indicate we'd like to send the request
# to our local model, not OpenAI's servers
openai.api_base = "http://127.0.0.1:8081"
openai.api_key = ""


def to_oai_chat(history):
    """Converts the gradio chat history format to
    the OpenAI chat history format:

    Gradio format: ['<user message>', '<bot message>']
    OpenAI format: [{'role': 'user', 'content': '<user message>'},
                    {'role': 'assistant', 'content': '<bot_message>'}]

    Additionally, this adds the 'system' message to the chat to tell the
    assistant how to act.
    """
    chat = [
        {
            "role": "system",
            "content": "You are a helpful AI Assistant who ends all of your responses with </BOT>",
        }
    ]

    for msg_pair in history:
        if msg_pair[0]:
            chat.append({"role": "user", "content": msg_pair[0]})
        if msg_pair[1]:
            chat.append({"role": "assistant", "content": msg_pair[1]})
    return chat


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=650)
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history):
        """Appends a submitted question to the history"""
        return "", history + [[user_message, None]]

    def bot(history):
        """Sends the chat history to our Llama-2 model server
        so that the model can respond appropriately
        """
        # Gradio chat -> OpenAI chat
        oai_chat = to_oai_chat(history)

        # Send chat history to our Llama-2 server
        response = openai.ChatCompletion.create(
            messages=oai_chat,
            stream=True,
            model="llama_2",
            temperature=0,
            # Used to stop runaway responses
            stop=["</BOT>"],
        )

        history[-1][1] = ""
        for response_chunk in response:
            # Filter through meta-data in the HTTP response to get response text
            next_token = response_chunk["choices"][0]["delta"].get("content")
            if next_token:
                history[-1][1] += next_token
                # Update the Gradio app with the streamed response
                yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch()
