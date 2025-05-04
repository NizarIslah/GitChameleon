# library: spacy
# version: 3.5.0
# extra_dependencies: ['numpy==1.26.4']
import spacy
from spacy.pipeline.span_ruler import SpanRuler


def get_labels(ruler: SpanRuler) -> tuple:
    """
    Get the labels of the SpanRuler.

    Args:
        ruler (SpanRuler): The SpanRuler to get the labels from.

    Returns:
        tuple: The labels of the SpanRuler.
    """
    return ruler.labels
