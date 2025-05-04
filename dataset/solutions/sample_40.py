# library: gradio
# version: 2.9.2
# extra_dependencies: ['black==22.1.0']
import gradio as gr


def process_image(image):
    return "Processed"


iface = gr.Interface(
    fn=process_image, inputs=gr.inputs.Image(), outputs=gr.outputs.Textbox()
)
