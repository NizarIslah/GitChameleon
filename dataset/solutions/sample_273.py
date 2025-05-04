# library: plotly
# version: 4.0.0
# extra_dependencies: ['chart-studio==1.0.0']
import plotly


def custom_api_usage() -> str:
    import chart_studio.api

    return chart_studio.api.__name__
