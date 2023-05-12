"""Detect bias by finding keywords related to protected concepts that rank high in LIME."""
from typing import Callable, List

from biaslyze.concept_detectors import KeywordConceptDetector
from biaslyze.evaluation_results import EvaluationResult
from biaslyze.evaluators import LimeBiasEvaluator


class LimeBiasDetector:
    """Detect bias by finding keywords related to protected concepts that rank high in LIME.

    Usage example:

        ```python
        from biaslyze.bias_detectors import LimeKeywordBiasDetector

        bias_detector = LimeKeywordBiasDetector(
            bias_evaluator=LimeBiasEvaluator(n_lime_samples=500),
            n_top_keywords=10
        )

        # detect bias in the model based on the given texts
        # here, clf is a scikit-learn text classification pipeline trained for a binary classification task
        detection_res = bias_detector.detect(
            texts=texts,
            predict_func=clf.predict_proba
        )

        # see a summary of the detection
        detection_res.summary()
        ```

    Attributes:
        n_top_keywords: In how many important LIME words should the method look for protected keywords.
        use_tokenizer: If keywords should only be searched in tokenized text. Can be useful for short keywords like 'she'.
        concept_detector: An instance of KeywordConceptDetector
        bias_evaluator: An instance of LimeBiasEvaluator
    """

    def __init__(
        self,
        n_top_keywords: int = 10,
        use_tokenizer: bool = False,
        concept_detector: KeywordConceptDetector = KeywordConceptDetector(),
        bias_evaluator: LimeBiasEvaluator = LimeBiasEvaluator(),
    ):
        self.n_top_keywords = n_top_keywords
        self.use_tokenizer = use_tokenizer
        self.concept_detector = concept_detector
        self.bias_evaluator = bias_evaluator

        # overwrite use_tokenizer
        self.concept_detector.use_tokenizer = self.use_tokenizer

    def detect(
        self, texts: List[str], predict_func: Callable[[List[str]], List[float]]
    ) -> EvaluationResult:
        """Detect bias using keyword concept detection and lime bias evaluation.

        Args:
            texts: List of texts to evaluate.
            predict_func: Function that predicts a for a given text. Currently only binary classification is supported.

        Returns:
            An EvaluationResults object containing the results.
        """
        detected_texts = self.concept_detector.detect(texts)

        evaluation_result = self.bias_evaluator.evaluate(
            predict_func=predict_func,
            texts=detected_texts,
            top_n=self.n_top_keywords,
        )

        return evaluation_result
