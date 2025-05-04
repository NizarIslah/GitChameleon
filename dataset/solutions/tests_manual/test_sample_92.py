import unittest
import spacy
from spacy.training import Example
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_92 import create_whitespace_variant

nlp = spacy.blank("en")

tokens = nlp("Hello world")
annotations = {"entities": [(0, 5, "GREETING")]}
example = Example.from_dict(tokens, annotations)

whitespace = " "
position = 1

augmented_example = create_whitespace_variant(nlp, example, whitespace, position)
expected_doc_annotation = {
    "cats": {},
    "entities": ["U-GREETING", "O", "O"],
    "spans": {},
    "links": {},
}

expected_token_annotation = {
    "ORTH": ["Hello", " ", "world"],
    "SPACY": [True, False, False],
    "TAG": ["", "", ""],
    "LEMMA": ["", "", ""],
    "POS": ["", "", ""],
    "MORPH": ["", "", ""],
    "HEAD": [0, 1, 2],
    "DEP": ["", "", ""],
    "SENT_START": [1, 0, 0],
}

assert augmented_example.to_dict()["doc_annotation"] == expected_doc_annotation
assert augmented_example.to_dict()["token_annotation"] == expected_token_annotation
