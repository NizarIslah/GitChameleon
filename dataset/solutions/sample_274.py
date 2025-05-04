# library: plotly
# version: 3.0.0
# extra_dependencies: []
import plotly.graph_objs as go


def custom_scatter(custom_color: str) -> go.Figure:
    return go.Figure(
        data=[go.Scatter(x=[0], y=[0], marker=go.scatter.Marker(color=custom_color))]
    )
