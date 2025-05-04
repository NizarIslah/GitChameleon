# library: plotly
# version: 5.10.0
# extra_dependencies: []
import plotly.graph_objects as go


def custom_fig(fig: go.Figure) -> go.Figure:
    return fig.update_layout(scene_camera=dict(eye=dict(x=1.25, y=1.25, z=1.25)))
