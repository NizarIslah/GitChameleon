# library: gradio
# version: 3.24.0
# extra_dependencies: []
import gradio as gr


def process_image(image):
    return "Processed"


iface = gr.Interface(fn=process_image, inputs=gr.Image(), outputs=gr.Label())
