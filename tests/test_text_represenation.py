"""This module contains tests for the text_representation module."""

from spacy.tokens import Doc
from spacy.vocab import Vocab

from biaslyze.text_representation import (
    TextRepresentation,
    Token,
    process_texts_with_spacy,
)


def test_token_init():
    """Test if a token is initialized correctly."""
    token = Token(text="she", start=0, end=3, whitespace_after=" ", shape="xxx")
    assert token.text == "she"
    assert token.start == 0
    assert token.end == 3
    assert token.whitespace_after == " "
    assert token.shape == "xxx"


def test_text_representation_init():
    """Test if a text representation is initialized correctly."""
    text_representation = TextRepresentation(
        text="She is a doctor",
        tokens=[
            Token(text="She", start=0, end=3, whitespace_after=" ", shape="Xxx"),
            Token(text="is", start=4, end=6, whitespace_after=" ", shape="xx"),
            Token(text="a", start=7, end=8, whitespace_after=" ", shape="x"),
            Token(text="doctor", start=9, end=15, whitespace_after="", shape="xxxxx"),
        ],
    )
    assert text_representation.text == "She is a doctor"
    assert len(text_representation.tokens) == 4
    assert text_representation.tokens[0].text == "She"
    assert text_representation.tokens[0].start == 0
    assert text_representation.tokens[0].end == 3
    assert text_representation.tokens[0].whitespace_after == " "
    assert text_representation.tokens[0].shape == "Xxx"


def test_text_representation_init_from_spacy():
    """Test if a text representation is initialized correctly from spacy doc."""
    spaces = [True, True, True, False]
    words = ["she", "is", "a", "doctor"]
    vocab = Vocab(strings=["she", "is", "a", "doctor"])
    doc = Doc(vocab, words=words, spaces=spaces)
    text_representation = TextRepresentation.from_spacy_doc(doc)
    assert text_representation.text == "she is a doctor"
    assert len(text_representation.tokens) == 4
    assert text_representation.tokens[0].text == "she"
    assert text_representation.tokens[0].start == 0
    assert text_representation.tokens[0].end == 3
    assert text_representation.tokens[0].whitespace_after == " "


def test_text_representation_contains():
    """Test if a text representation contains a string."""
    text_representation = TextRepresentation(
        text="She is a doctor",
        tokens=[
            Token(text="She", start=0, end=3, whitespace_after=" ", shape="Xxx"),
            Token(text="is", start=4, end=6, whitespace_after=" ", shape="xx"),
            Token(text="a", start=7, end=8, whitespace_after=" ", shape="x"),
            Token(text="doctor", start=9, end=15, whitespace_after="", shape="xxxxx"),
        ],
    )
    assert "She" in text_representation
    assert "He" not in text_representation


def test_process_texts_with_spacy():
    """Test if processing texts with spacy works as expected."""
    texts = ["she is a doctor"]
    text_representation = process_texts_with_spacy(texts=texts)
    assert text_representation[0].text == "she is a doctor"
    assert len(text_representation[0].tokens) == 4
    assert text_representation[0].tokens[0].text == "she"
    assert text_representation[0].tokens[0].start == 0
    assert text_representation[0].tokens[0].end == 3
