# library: plotly
# version: 4.0.0
# extra_dependencies: []
import plotly
import plotly.graph_objects as go


def custom_figure(x_data: list[int], y_data: list[int]) -> go.Figure:
    import plotly.graph_objects

    fig = plotly.graph_objects.Figure()
    fig.add_trace(plotly.graph_objects.Scatter(x=x_data, y=y_data))
    return fig
