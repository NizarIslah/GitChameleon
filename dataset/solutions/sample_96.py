# library: nltk
# version: 3.7
# extra_dependencies: []
import nltk

nltk.download("sinica_treebank")
from nltk.tree import Tree
from nltk.corpus import sinica_treebank


def parse_sinica_treebank_sentence(sentence: str) -> Tree:
    """
    Parse a sentence from the Sinica Treebank.

    Args:
        sentence (str): The sentence to parse.

    Returns:
        Tree: The parsed tree.
    """
    return Tree.fromstring(sentence)
