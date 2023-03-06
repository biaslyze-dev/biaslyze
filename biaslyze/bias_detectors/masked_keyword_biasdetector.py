"""Detect bias by finding keywords related to protected concepts with a language model that change the prediction a lot."""
from typing import List, Callable

from biaslyze.concept_detectors import KeywordConceptDetector
from biaslyze.evaluators import (
    MaskedLMBiasEvaluator,
)
from biaslyze.evaluation_results import EvaluationResult


class MaskedKeywordBiasDetector:
    """Detect bias by finding keywords related to protected concepts with a language model that change the prediction a lot."""

    def __init__(
        self,
        predict_func: Callable[[List[str]], List[float]],
        n_resample_keywords: int = 10,
        concept_detector=KeywordConceptDetector(),
        bias_evaluator=MaskedLMBiasEvaluator(),
    ):
        self.predict_func = predict_func
        self.n_resample_keywords = n_resample_keywords
        self.concept_detector = concept_detector
        self.bias_evaluator = bias_evaluator

    def detect(self, texts: List[str]) -> EvaluationResult:
        detected_texts = self.concept_detector.detect(texts)

        evaluation_result = self.bias_evaluator.evaluate(
            predict_func=self.predict_func,
            texts=detected_texts,
            n_resample_keywords=self.n_resample_keywords,
        )

        return evaluation_result
