"""Tests for counterfactual_bias_detector.py"""

import numpy as np
import pytest

from biaslyze.bias_detectors import CounterfactualBiasDetector
from biaslyze.concept_class import Concept


def test_process():
    """Test process method of CounterfactualBiasDetector"""
    detector = CounterfactualBiasDetector()

    with pytest.raises(ValueError):
        detector.process(texts=["text1", "text2"], predict_func=None)

    with pytest.raises(ValueError):
        detector.process(texts=None, predict_func=lambda x: x)

    with pytest.raises(ValueError):
        detector.process(texts=None, predict_func=None)

    with pytest.raises(ValueError):
        detector.process(
            texts=["text1", "text2"],
            predict_func=lambda x: x,
            max_counterfactual_samples=-1,
        )

    with pytest.raises(ValueError):
        detector.process(
            texts=["text1", "text2"],
            predict_func=lambda x: x,
            max_counterfactual_samples="abc",
        )

    with pytest.raises(ValueError):
        detector.process(
            texts=["text1", "text2"],
            predict_func=lambda x: x,
            concepts_to_consider="abc",
        )

    with pytest.raises(ValueError):
        detector.process(
            texts=["text1", "text2"],
            predict_func=lambda x: x,
            concepts_to_consider=["abc"],
            max_counterfactual_samples=-1,
        )

    with pytest.raises(ValueError):
        detector.process(
            texts=["text1", "text2"],
            predict_func=lambda x: x,
            concepts_to_consider=["abc"],
            max_counterfactual_samples="abc",
        )


def test_init_detector():
    """Test init method of CounterfactualBiasDetector"""
    detector = CounterfactualBiasDetector(use_tokenizer=True)

    assert detector.use_tokenizer == True
    assert detector.concept_detector.use_tokenizer == True


def test_process_positive():
    """ "Test process method of CounterfactualBiasDetector with a successful run"""
    detector = CounterfactualBiasDetector(use_tokenizer=True)

    # test with a successful run
    res = detector.process(
        texts=["woman", "man"],
        predict_func=lambda x: np.array([[0.1, 0.9], [0.9, 0.1]]),
        concepts_to_consider=["gender"],
        max_counterfactual_samples_per_text=1,
    )

    assert res is not None
    assert len(res._get_counterfactual_samples_by_concept("gender")) == 2


def test_register_concepts():
    """Test the register concepts method of CounterfactualBiasDetector."""
    bias_detector = CounterfactualBiasDetector(lang="de")
    n_concepts = len(bias_detector.concepts)
    names_concept = Concept.from_dict_keyword_list(
        name="names",
        lang="de",
        keywords=[{"keyword": "Hans", "function": ["name"]}],
    )
    bias_detector.register_concept(names_concept)

    assert len(bias_detector.concepts) == n_concepts + 1
