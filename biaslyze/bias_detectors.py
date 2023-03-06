"""Contains bias detector pipelines combining concept detectors and bias evaluators."""
from typing import List, Callable

from biaslyze.concept_detectors import KeywordConceptDetector
from biaslyze.evaluators import LimeBiasEvaluator


class KeywordBiasDetector:
    def __init__(
        self,
        predict_func: Callable[[List[str]], List[float]],
        n_top_keywords: int = 10,
        concept_detector=KeywordConceptDetector(),
        bias_evaluator=LimeBiasEvaluator(),
    ):
        self.predict_func = predict_func
        self.n_top_keywords = n_top_keywords
        self.concept_detector = concept_detector
        self.bias_evaluator = bias_evaluator

    def detect(self, texts: List[str]):
        detected_texts = self.concept_detector.detect(texts)

        biased_samples = self.bias_evaluator.evaluate(
            predict_func=self.predict_func,
            texts=detected_texts,
            top_n=self.n_top_keywords,
        )

        return biased_samples
