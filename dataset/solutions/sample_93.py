# library: spacy
# version: 3.5.0
# extra_dependencies: ['numpy==1.26.4']
import spacy
from spacy.pipeline.span_ruler import SpanRuler


def remove_pattern_by_id(ruler: SpanRuler, pattern_id: str) -> None:
    """
    Remove a pattern from the SpanRuler by its ID.

    Args:
        ruler (SpanRuler): The SpanRuler to remove the pattern from.
        pattern_id (str): The ID of the pattern to remove.

    Returns:
        None
    """
    ruler.remove_by_id(pattern_id)
