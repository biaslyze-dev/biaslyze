"""Tests for functions to help calculate counterfactual bias metrics."""
import numpy as np

from biaslyze.bias_detectors.counterfactual_biasdetector import (
    _calculate_counterfactual_scores,
    _extract_counterfactual_concept_samples,
)
from biaslyze.concept_class import Concept, Keyword


# create a mock token class with a text attribute
class MockToken:
    def __init__(self, text):
        self.text = text
        self.whitespace_ = " "


# mock the tokenizer with a pipe method
class MockTokenizer:
    def pipe(self, texts):
        return [[MockToken(text=t) for t in text.split(" ")] for text in texts]


def test_extract_counterfactual_concept_samples():
    """Test _extract_counterfactual_concept_samples"""
    concept = Concept(
        name="gender",
        lang="en",
        keywords=[
            Keyword(text="she", functions=["PRON"], category="female"),
            Keyword(text="he", functions=["PRON"], category="male"),
            Keyword(text="they", functions=["PRON"], category="neutral"),
        ],
    )
    texts = [
        "she is a doctor",
        "he is a nurse",
        "they are a doctor",
        "they are a nurse",
    ]

    counterfactual_samples = _extract_counterfactual_concept_samples(
        concept=concept,
        texts=texts,
    )

    assert len(counterfactual_samples) == len(texts) * len(concept.keywords)


def test_calculate_counterfactual_score():
    """Test _calculate_counterfactual_scores."""

    concept = Concept(
        name="gender",
        lang="en",
        keywords=[
            Keyword(text="she", functions=["PRON"], category="female"),
            Keyword(text="he", functions=["PRON"], category="male"),
            Keyword(text="they", functions=["PRON"], category="neutral"),
        ],
    )
    texts = [
        "she is a doctor",
        "he is a nurse",
        "they are a doctor",
        "they are a nurse",
    ]

    counterfactual_samples = _extract_counterfactual_concept_samples(
        concept=concept,
        texts=texts,
    )

    predict_func = lambda x: np.array(
        [[0.1, 0.9] if ("she" in xi) else [0.9, 0.1] for xi in x]
    )

    scores = _calculate_counterfactual_scores(
        bias_keyword="she",
        samples=counterfactual_samples,
        predict_func=predict_func,
    )

    assert len(scores) == len(texts)
    assert scores[0] == 0.0
    assert scores[1] == 0.8
    assert scores[2] == 0.8
    assert scores[3] == 0.8
