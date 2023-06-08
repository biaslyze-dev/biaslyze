"""This module contains tests for the Concept class in the biaslyze package."""

from biaslyze.concept_class import Concept, Keyword
from biaslyze.text_representation import TextRepresentation, Token


def test_keyword_init():
    """Test the Keyword class constructor."""
    keyword = Keyword(text="she", function=["subject"], category="pronoun")
    assert keyword.text == "she"
    assert keyword.function == ["subject"]
    assert keyword.category == "pronoun"


def test_keyword_can_replace_token():
    """Test the Keyword class can_replace_token method."""
    keyword = Keyword(text="she", function=["subject"], category="pronoun")
    assert keyword.can_replace_token(
        token=Token(text="she", start=11, end=14, whitespace_after=" ")
    )


def test_keyword_equal_to_token():
    """Test the Keyword class equal_to_token method."""
    keyword = Keyword(text="she", function=["subject"], category="pronoun")
    assert keyword.equal_to_token(
        token=Token(text="she", start=0, end=3, whitespace_after=" ")
    )
    assert not keyword.equal_to_token(
        token=Token(text="he", start=0, end=2, whitespace_after=" ")
    )


def test_keyword_get_keyword_in_style_of_token():
    """Test the Keyword class get_keyword_in_style_of_token method."""
    keyword = Keyword(text="she", function=["subject"], category="pronoun")
    assert (
        keyword.get_keyword_in_style_of_token(
            token=Token(text="He", start=0, end=3, whitespace_after=" ")
        )
        == "she"
    )


def test_concept_init():
    """Test the Concept class constructor."""
    concept = Concept(
        name="gender",
        keywords=[
            Keyword(text="she", function=["subject"], category="pronoun"),
            Keyword(text="he", function=["subject"], category="pronoun"),
            Keyword(text="her", function=["object"], category="pronoun"),
        ],
    )
    assert len(concept.keywords) == 3
    assert concept.name == "gender"
    assert concept.keywords[0].text == "she"
    assert concept.keywords[1].text == "he"
    assert concept.keywords[2].text == "her"
    assert concept.keywords[0].function == ["subject"]
    assert concept.keywords[1].function == ["subject"]
    assert concept.keywords[2].function == ["object"]
    assert concept.keywords[0].category == "pronoun"
    assert concept.keywords[1].category == "pronoun"
    assert concept.keywords[2].category == "pronoun"


def test_concept_get_present_keywords():
    """Test the Concept class get_present_keywords method."""
    concept = Concept(
        name="gender",
        keywords=[
            Keyword(text="she", function=["subject"], category="pronoun"),
            Keyword(text="he", function=["subject"], category="pronoun"),
            Keyword(text="her", function=["object"], category="pronoun"),
        ],
    )

    text_representation = TextRepresentation(
        text="She is a doctor.",
        tokens=[
            Token(text="She", start=0, end=3, whitespace_after=" "),
            Token(text="is", start=4, end=6, whitespace_after=" "),
            Token(text="a", start=7, end=8, whitespace_after=" "),
            Token(text="doctor", start=9, end=15, whitespace_after=""),
            Token(text=".", start=15, end=16, whitespace_after=""),
        ],
    )

    present_keywords = concept.get_present_keywords(
        text_representation=text_representation
    )
    assert len(present_keywords) == 1
    assert present_keywords[0].text == "she"


def test_concept_get_counterfactual_texts():
    """Test the Concept class get_counterfactual_texts method."""
    text_representation = TextRepresentation(
        text="She is a doctor.",
        tokens=[
            Token(text="She", start=0, end=3, whitespace_after=" "),
            Token(text="is", start=4, end=6, whitespace_after=" "),
            Token(text="a", start=7, end=8, whitespace_after=" "),
            Token(text="doctor", start=9, end=15, whitespace_after=""),
            Token(text=".", start=15, end=16, whitespace_after=""),
        ],
    )

    keyword = Keyword(text="she", function=["subject"], category="pronoun")

    concept = Concept(
        name="gender",
        keywords=[
            Keyword(text="he", function=["subject"], category="pronoun"),
            Keyword(text="her", function=["object"], category="pronoun"),
            Keyword(text="his", function=["possessive"], category="pronoun"),
            Keyword(text="him", function=["object"], category="pronoun"),
        ],
    )

    counterfactual_texts = concept.get_counterfactual_texts(
        text_representation=text_representation,
        keyword=keyword,
        n_texts=2,
    )

    assert len(counterfactual_texts) == 2


def test_concept_from_dict_keyword_list():
    """Test the Concept class from_dict_keyword_list class method."""
    concept_name = "gender"
    keyword_list = [
        {"keyword": "she", "function": ["subject"], "category": "pronoun"},
        {"keyword": "he", "function": ["subject"], "category": "pronoun"},
        {"keyword": "her", "function": ["object"], "category": "pronoun"},
        {"keyword": "his", "function": ["possessive"], "category": "pronoun"},
    ]

    concept = Concept.from_dict_keyword_list(name=concept_name, keywords=keyword_list)

    assert len(concept.keywords) == 4
    assert concept.name == concept_name
