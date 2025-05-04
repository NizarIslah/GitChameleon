# library: gradio
# version: 3.17.0
# extra_dependencies: []
import gradio as gr


def get_selected_options(options):
    return f"Selected options: {options}"


selection_options = ["angola", "pakistan", "canada"]

iface = gr.Interface(
    get_selected_options,
    inputs=gr.Dropdown(selection_options, multiselect=True),
    outputs="text",
)
