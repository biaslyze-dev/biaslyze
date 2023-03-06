"""Contains bias detector pipelines combining concept detectors and bias evaluators."""
from typing import List, Callable

from biaslyze.concept_detectors import KeywordConceptDetector
from biaslyze.evaluators import LimeBiasEvaluator


class KeywordBiasDetector:
    def __init__(
        self,
        predict_func: Callable[[List[str]], List[float]],
        concept_detector=KeywordConceptDetector(),
        bias_evaluator=LimeBiasEvaluator(),
    ):
        self.predict_func = predict_func
        self.concept_detector = concept_detector
        self.bias_evaluator = bias_evaluator

    def detect(self, texts: List[str], labels: List):
        detected_texts, detected_labels = self.concept_detector.detect(texts, labels)

        biased_samples = self.bias_evaluator.evaluate(
            self.predict_func, detected_texts, detected_labels
        )

        return biased_samples
