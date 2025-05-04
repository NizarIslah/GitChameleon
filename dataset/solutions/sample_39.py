# library: gradio
# version: 3.0.0
# extra_dependencies: []
import gradio as gr


def display_image():
    return "https://image_placeholder.com/42"


iface = gr.Interface(fn=display_image, inputs=[], outputs=gr.Image())
