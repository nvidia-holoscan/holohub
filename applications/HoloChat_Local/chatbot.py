from llm import LLM
import gradio as gr

initial_prompt = "Welcome to HoloChat! How can I assist you today?"


def ask_question(message, chat_history):
    if chat_history is None:
        return "", [[None, initial_prompt]]
    if chat_history[-1][0] == initial_prompt:
        chat_history[-1][1] = message
        return "", chat_history

    return "", chat_history + [[message, None]]


def stream_response(chat_history, llm):
    if llm is None:
        llm = LLM()

    response = llm.answer_question(chat_history)

    for chunk in response:
        yield chunk, llm


def reset_textbox():
    return gr.update(value="")


def set_visible_false():
    return gr.update(visible=False)


def set_visible_true():
    return gr.update(visible=True)


def main():
    logo_size = 92

    title = "HoloChat"
    theme = gr.themes.Soft(text_size=gr.themes.sizes.text_md).set(
        button_primary_background_fill="#76b900",
        button_primary_background_fill_dark="#AAAAAA",
    )

    with gr.Blocks(
        css="""#col_container { margin-left: auto; margin-right: auto;} #chatbot {height: 740px; overflow: auto;}""",
        theme=theme,
        title=title,
    ) as demo:
        llm = gr.State()
        with gr.Row(variant="compact"):
            image1 = gr.Image(
                "holoscan.png", show_label=False, height=logo_size, width=logo_size
            )

        with gr.Group(visible=True):
            with gr.Row():
                with gr.Column(scale=60):
                    chatbot = gr.Chatbot(
                        value=[(None, initial_prompt)],
                        label="HoloChat",
                        elem_id="chatbot",
                    )
            with gr.Row():
                with gr.Column(scale=8):
                    tbInput = gr.Textbox(
                        placeholder="What questions do you have for HoloChat?",
                        lines=1,
                        label="Type an input and press Enter",
                    )
                with gr.Column(scale=2):
                    btnChat = gr.Button(scale=2)

            with gr.Accordion(label="Examples", open=True):
                gr.Examples(
                    examples=[
                        ["What operating system can I use with the Holoscan SDK?"],
                        ["What hardware does Holoscan support?"],
                        ["How do I create a C++ Holoscan Operator?"],
                        [
                            "Create a Python Holoscan 'hello world' app with video as input, use HoloViz to print 'Hello World' on each frame, and then output it to the user. After the code explain the process step-by-step."
                        ],
                    ],
                    inputs=tbInput,
                )

        tbInput.submit(
            ask_question, [tbInput, chatbot], [tbInput, chatbot], queue=False
        ).then(stream_response, [chatbot, llm], [chatbot, llm])
        btnChat.click(
            ask_question, [tbInput, chatbot], [tbInput, chatbot], queue=False
        ).then(stream_response, [chatbot, llm], [chatbot, llm])

    demo.queue(max_size=99, concurrency_count=20).launch(
        server_name="0.0.0.0", debug=True, share=True
    )


if __name__ == "__main__":
    main()
