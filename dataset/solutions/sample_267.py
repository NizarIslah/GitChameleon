# library: plotly
# version: 5.8.0
# extra_dependencies: []
import plotly.graph_objects as go


def custom_fig(fig: go.Figure) -> go.Figure:
    return fig.add_annotation(
        x=0.5,
        y=0.5,
        text="Example Annotation",
        xref="paper",
        yref="paper",
        showarrow=False,
    )
