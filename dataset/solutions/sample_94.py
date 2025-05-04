# library: nltk
# version: 3.7
# extra_dependencies: []
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet


def align_words_func(hypothesis, reference):
    """
    Align words between hypothesis and reference sentences.

    Args:
        hypothesis (list): List of words in the hypothesis sentence.
        reference (list): List of words in the reference sentence.

    Returns:
        tuple: A tuple containing the aligned matches, unmatched hypothesis, and unmatched reference.
    """
    return nltk.translate.meteor_score.align_words(hypothesis, reference)
