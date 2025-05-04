# library: plotly
# version: 5.10.0
# extra_dependencies: []
import plotly.graph_objects as go


def custom_fig(x_data: list[int], y_data: list[int], color_set: str) -> go.Figure:
    return go.Figure(data=go.Scatter(x=x_data, y=y_data, error_y=dict(color=color_set)))
