# library: plotly
# version: 4.8.0
# extra_dependencies: []
import plotly.graph_objects as go


def custom_fig(x_data: list[str], y_data: list[int]) -> go.Figure:
    return go.Figure(data=[go.Bar(x=x_data, y=y_data, orientation="v")])
