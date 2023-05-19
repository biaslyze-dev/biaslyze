"""Detect bias by finding keywords related to protected concepts that rank high in LIME."""
import warnings
from typing import Callable, List

import numpy as np
from eli5.lime import TextExplainer
from loguru import logger
from tqdm import tqdm

from biaslyze.concept_detectors import KeywordConceptDetector
from biaslyze.concepts import CONCEPTS
from biaslyze.results.lime_detection_results import (
    LimeDetectionResult,
    LimeSampleResult,
)


class LimeBiasDetector:
    """Detect bias by finding keywords related to protected concepts that rank high in LIME.

    Usage example:

        ```python
        from biaslyze.bias_detectors import LimeKeywordBiasDetector

        bias_detector = LimeKeywordBiasDetector(
            n_lime_samples=500,
        )

        # detect bias in the model based on the given texts
        # here, clf is a scikit-learn text classification pipeline trained for a binary classification task
        detection_res = bias_detector.detect(
            texts=texts,
            predict_func=clf.predict_proba,
            n_top_keywords=10,
        )

        # see a summary of the detection
        detection_res.summary()
        ```

    Attributes:
        n_lime_samples: Number of perturbed samples to create for each LIME run.
        use_tokenizer: If keywords should only be searched in tokenized text. Can be useful for short keywords like 'she'.
        concept_detector: An instance of KeywordConceptDetector
    """

    def __init__(
        self,
        n_lime_samples: int = 1000,
        use_tokenizer: bool = False,
        concept_detector: KeywordConceptDetector = KeywordConceptDetector(),
    ):
        self.use_tokenizer = use_tokenizer
        self.concept_detector = concept_detector

        # overwrite use_tokenizer
        self.concept_detector.use_tokenizer = self.use_tokenizer

        # LIME configuration
        self.n_lime_samples = n_lime_samples
        self.explainer = TextExplainer(n_samples=n_lime_samples)
        # only use unigrams
        self.explainer.vec.ngram_range = (1, 1)

    def detect(
        self,
        texts: List[str],
        predict_func: Callable[[List[str]], List[float]],
        top_n_keywords: int = 10,
    ) -> LimeDetectionResult:
        """Detect bias using keyword concept detection and lime bias evaluation.

        Args:
            texts: List of texts to evaluate.
            predict_func: Function that predicts a for a given text. Currently only binary classification is supported.
            top_n_keywords: How many keywords detected by LIME should be considered for bias detection.

        Returns:
            A LimeDetectionResult containing all samples with detected bias.
        """
        warnings.filterwarnings("ignore", category=FutureWarning)
        detected_texts = self.concept_detector.detect(texts)
        logger.info(f"Started bias detection on {len(detected_texts)} samples...")
        biased_samples = []
        for text in tqdm(texts):
            # use LIME on the given text sample
            self.explainer.fit(text, predict_func)
            # get the explanation from LIME (linear model coefficients and feature names)
            interpret_sample_dict = {
                np.sign(coef)
                * np.abs(coef)
                / sum(np.abs(self.explainer.clf_.coef_[0])): token
                for coef, token in zip(
                    self.explainer.clf_.coef_[0],
                    self.explainer.vec_.get_feature_names_out(),
                )
            }
            # get the most important tokens from the explanation
            top_interpret_sample_dict = sorted(
                interpret_sample_dict.items(), key=lambda x: -np.abs(x[0])
            )[: min(len(interpret_sample_dict), top_n_keywords)]
            important_tokens = [w.lower() for (_, w) in top_interpret_sample_dict]
            token_scores = [c for (c, _) in top_interpret_sample_dict]

            # check for concepts reasons
            bias_indicator_tokens = []
            bias_concepts = []
            for concept, concept_keywords in CONCEPTS.items():
                biased_tokens_set = set(
                    [keyword_dict.get("keyword") for keyword_dict in concept_keywords]
                ).intersection(set(important_tokens))
                if len(biased_tokens_set) > 0:
                    bias_concepts.append(concept)
                    bias_indicator_tokens.extend(list(biased_tokens_set))

            if len(bias_concepts) > 0:
                biased_samples.append(
                    LimeSampleResult(
                        text=text,
                        bias_concepts=bias_concepts,
                        bias_reasons=bias_indicator_tokens,
                        top_words=important_tokens,
                        num_tokens=len(interpret_sample_dict),
                        keyword_position=min(
                            [
                                important_tokens.index(bias_token)
                                for bias_token in bias_indicator_tokens
                            ]
                        ),
                        score=max(
                            [
                                token_scores[important_tokens.index(bias_token)]
                                for bias_token in bias_indicator_tokens
                            ]
                        ),
                        metrics=self.explainer.metrics_,
                    )
                )

        return LimeDetectionResult(biased_samples)
