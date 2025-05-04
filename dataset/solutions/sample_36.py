# library: gradio
# version: 3.24.0
# extra_dependencies: []
import gradio as gr


def render_quadratic_formula():
    pass


interface = gr.Interface(fn=render_quadratic_formula, inputs=[], outputs="text")


def render_quadratic_formula():
    formula = "$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$"
    return formula
