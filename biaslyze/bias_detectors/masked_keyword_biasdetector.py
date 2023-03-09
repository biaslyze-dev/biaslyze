"""Detect bias by finding keywords related to protected concepts with a language model that change the prediction a lot."""
from typing import Callable, List

from biaslyze.concept_detectors import KeywordConceptDetector
from biaslyze.evaluation_results import EvaluationResult
from biaslyze.evaluators import MaskedLMBiasEvaluator


class MaskedKeywordBiasDetector:
    """Detect bias by finding keywords related to protected concepts with a language model that change the prediction a lot.

    Usage example:

        ```python
        from biaslyze.bias_detectors import MaskedKeywordBiasDetector

        bias_detector = MaskedKeywordBiasDetector(
            n_resample_keywords=10
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
        n_resample_keywords: How many time replace a found keyword by different concept keywords.
        use_tokenizer: If keywords should only be searched in tokenized text. Can be useful for short keywords like 'she'.
        concept_detector: an instance of KeywordConceptDetector
        bias_evaluator: an instance of LimeBiasEvaluator
    """

    def __init__(
        self,
        n_resample_keywords: int = 10,
        use_tokenizer: bool = False,
        concept_detector: KeywordConceptDetector = KeywordConceptDetector(),
        bias_evaluator: MaskedLMBiasEvaluator = MaskedLMBiasEvaluator(),
    ):
        self.n_resample_keywords = n_resample_keywords
        self.use_tokenizer = use_tokenizer
        self.concept_detector = concept_detector
        self.bias_evaluator = bias_evaluator

        # overwrite use_tokenizer
        self.concept_detector.use_tokenizer = self.use_tokenizer

    def detect(
        self, texts: List[str], predict_func: Callable[[List[str]], List[float]]
    ) -> EvaluationResult:
        """Detect bias by masking out words.

        Args:
            texts: texts to probe the model for bias.
            predict_func: Function to run the texts through the model and get probabilities as outputs.
        """

        detected_texts = self.concept_detector.detect(texts)

        evaluation_result = self.bias_evaluator.evaluate(
            predict_func=predict_func,
            texts=detected_texts,
            n_resample_keywords=self.n_resample_keywords,
        )

        return evaluation_result
