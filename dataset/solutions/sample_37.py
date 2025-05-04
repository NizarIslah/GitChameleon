# library: gradio
# version: 3.36.0
# extra_dependencies: []
import gradio as gr


def render_quadratic_formula():
    formula = "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}"
    return formula


interface = gr.Chatbot(fn=render_quadratic_formula, latex_delimiters=("$$", "$$"))
