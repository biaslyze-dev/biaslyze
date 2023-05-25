"""Tests for functions to help calculate counterfactual bias metrics."""
from biaslyze.bias_detectors.counterfactual_biasdetector import (
    _extract_counterfactual_concept_samples,
)
from biaslyze.concepts import CONCEPTS


def test_extract_counterfactual_concept_samples():
    """Test _extract_counterfactual_concept_samples"""
    concept = "gender"
    texts = ["she is a doctor", "he is a nurse"]

    # create a mock token class with a text attribute
    class MockToken:
        def __init__(self, text):
            self.text = text
            self.whitespace_ = " "

    # mock the tokenizer with a pipe method
    class MockTokenizer:
        def pipe(self, texts):
            return [[MockToken(text=t) for t in text.split(" ")] for text in texts]

    tokenizer = MockTokenizer()

    counterfactual_samples = _extract_counterfactual_concept_samples(
        concept=concept,
        texts=texts,
        tokenizer=tokenizer,
    )

    assert len(counterfactual_samples) == len(texts) * len(CONCEPTS[concept])

