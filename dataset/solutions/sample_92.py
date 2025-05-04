# library: spacy
# version: 3.5.0
# extra_dependencies: ['numpy==1.26.4']
import spacy
from spacy.training import Example
from spacy.training import augment


def create_whitespace_variant(
    nlp: spacy.Language, example: Example, whitespace: str, position: int
) -> Example:
    """
    Create a whitespace variant of the given example.

    Args:
        nlp (Language): The spaCy language model.
        example (Example): The example to augment.
        whitespace (str): The whitespace to insert.
        position (int): The position to insert the whitespace.

    Returns:
        Example: The augmented example.
    """
    return augment.make_whitespace_variant(nlp, example, whitespace, position)
