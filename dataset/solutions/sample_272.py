# library: plotly
# version: 4.0.0
# extra_dependencies: ['chart-studio==1.0.0']
import plotly


def custom_chart_studio_usage() -> bool:
    import chart_studio.plotly

    return hasattr(chart_studio.plotly, "plot")
