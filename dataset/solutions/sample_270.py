# library: plotly
# version: 4.0.0
# extra_dependencies: []
import plotly
import plotly.graph_objects as go


def custom_make_subplots(rows: int, cols: int) -> go.Figure:
    return plotly.subplots.make_subplots(rows=rows, cols=cols)
