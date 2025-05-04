# library: nltk
# version: 3.7
# extra_dependencies: []
import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk.corpus import wordnet


def get_synset_examples(synset: str) -> list:
    """
    Get examples for a given synset.

    Args:
        synset (str): The synset to get examples for.

    Returns:
        list: A list of examples for the synset.
    """
    return wordnet.synset(synset).examples()
