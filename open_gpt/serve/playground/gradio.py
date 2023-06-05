import os

import gradio as gr
import mdtex2html

from ...logs import logger
from .gradio_chatbot import Chatbot
from .gradio_css import code_highlight_css

base_css = (
    code_highlight_css
    + """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
"""
)

title_markdown = """
# ☄️ Chat with Large-scale Multimodal Models

[[Project Page]](https://opengpt.github.io) [[Code]](https://github.com/jinaai/opengpt)
"""

_get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def _load(url_params, request: gr.Request):
    logger.info(f"loading playground. ip: {request.client.host}. params: {url_params}")

    # dropdown_update = gr.Dropdown.update(visible=True)
    return (
        {},
        gr.Chatbot.update(visible=True),
        gr.Textbox.update(visible=True),
        gr.Button.update(visible=True),
        gr.Row.update(visible=True),
        gr.Accordion.update(visible=True),
    )


def create_playground(embed_mode: bool = False):
    textbox = gr.Textbox(
        show_label=False,
        placeholder="Enter textual instructions and press ENTER",
        visible=False,
    ).style(container=False)

    with gr.Blocks(
        title=__package__, theme=gr.themes.Base(), css=base_css
    ) as playground:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):
                imagebox = gr.Image(type="pil")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad"],
                    value="Crop",
                    label="Preprocess for non-square image",
                )

                cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(
                    examples=[
                        [
                            f"{cur_dir}/examples/extreme_ironing.jpeg",
                            "What is unusual about this image?",
                        ],
                        [
                            f"{cur_dir}/examples/waterview.jpeg",
                            "What are the things I should be cautious about when I visit here?",
                        ],
                    ],
                    inputs=[imagebox, textbox],
                )

                with gr.Accordion(
                    "Parameters", open=False, visible=False
                ) as parameter_row:
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.2,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    max_output_tokens = gr.Slider(
                        minimum=0,
                        maximum=1024,
                        value=512,
                        step=64,
                        interactive=True,
                        label="Max output tokens",
                    )

            with gr.Column(scale=6):
                chatbot = Chatbot(
                    elem_id="chatbot", label="Chatbot", visible=False
                ).style(height=550)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=60):
                        submit_btn = gr.Button(value="Submit", visible=False)
                with gr.Row(visible=False) as button_row:
                    # upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    # downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    # flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

        url_params = gr.JSON(visible=False)

        playground.load(
            _load,
            [url_params],
            [state, chatbot, textbox, submit_btn, button_row, parameter_row],
            _js=_get_window_url_params,
        )
    return playground
