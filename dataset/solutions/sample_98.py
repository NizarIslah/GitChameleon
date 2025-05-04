# library: nltk
# version: 3.5
# extra_dependencies: []
import nltk.tokenize.destructive


def tokenize_sentence(sentence: str) -> list:
    """
    Tokenize a sentence into words.

    Args:
        sentence (str): The sentence to tokenize.

    Returns:
        list: A list of tokens.
    """
    return nltk.tokenize.destructive.NLTKWordTokenizer().tokenize(sentence)
