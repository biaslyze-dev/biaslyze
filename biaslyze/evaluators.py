"""Contains classes to evaluate the bias of detected concepts."""
import random
import warnings
from typing import List

import numpy as np
from eli5.lime import TextExplainer
from tqdm import tqdm
from transformers import pipeline

from biaslyze.concepts import CONCEPTS
from biaslyze.evaluation_results import BiasedSampleResult, EvaluationResult


class LimeBiasEvaluator:
    """Evaluate bias in text based on LIME.
    
    Attributes:
        n_lime_samples: Number of perturbed samples to create for each LIME run.
    """
    def __init__(self, n_lime_samples: int = 100):
        self.n_lime_samples = n_lime_samples
        self.explainer = TextExplainer(n_samples=n_lime_samples)

    def evaluate(
        self, predict_func, texts: List[str], top_n: int = 10
    ) -> EvaluationResult:
        """Evaluate if a bias is present with LIME.
        
        Args:
            predict_func: Function that predicts a for a given text. Currently only binary classification is supported.
            texts: List of texts to evaluate.
            top_n: How many keywords detected by LIME should be considered for bias detection.

        Returns:
            EvaluationResult object containing information on the detected bias.
        """
        warnings.filterwarnings("ignore", category=FutureWarning)
        biased_samples = []
        for text in tqdm(texts):
            self.explainer.fit(text, predict_func)
            interpret_sample_dict = {
                coef: token
                for coef, token in zip(
                    self.explainer.clf_.coef_[0],
                    self.explainer.vec_.get_feature_names_out(),
                )
            }
            top_interpret_sample_dict = sorted(
                interpret_sample_dict.items(), key=lambda x: -np.abs(x[0])
            )[: min(len(interpret_sample_dict), top_n)]
            important_tokens = [w.lower() for (_, w) in top_interpret_sample_dict]

            # check for concepts reasons
            bias_indicator_tokens = []
            bias_concepts = []
            for concept, concept_keywords in CONCEPTS.items():
                biased_tokens_set = set(concept_keywords).intersection(
                    set(important_tokens)
                )
                if len(biased_tokens_set) > 0:
                    bias_concepts.append(concept)
                    bias_indicator_tokens.extend(list(biased_tokens_set))

            if len(bias_concepts) > 0:
                biased_samples.append(
                    BiasedSampleResult(text, bias_concepts, bias_indicator_tokens)
                )

        return EvaluationResult(biased_samples)


class MaskedLMBiasEvaluator:
    def __init__(
        self,
    ):
        self._lm = pipeline("fill-mask", model="distilbert-base-uncased")

    def evaluate(
        self, predict_func, texts: List[str], n_resample_keywords: int = 10
    ) -> EvaluationResult:
        """Evaluate if a bias is present with masked language models.

        Use a masked language model to resample keywords in texts and measure the difference in prediction.
        If the difference is 'large' we call is biased sample.

        Note:
            The language model might contain bias itself and resampling might be not diverse.
        """

        biased_samples = []

        for text in tqdm(texts):
            bias_concepts = []
            bias_indicator_tokens = []
            for concept, concept_keywords in CONCEPTS.items():
                probabilities = []
                # TODO: this is to simple, use spacy for tokenization
                present_keywords = list(
                    keyword for keyword in concept_keywords if keyword in text.lower()
                )
                if not present_keywords:
                    continue
                for _ in range(n_resample_keywords):
                    mask_keyword = random.choice(present_keywords)

                    # resample with the language model
                    masked_text = (
                        text[: text.lower().find(mask_keyword)]
                        + "[MASK]"
                        + text[text.lower().find(mask_keyword) + len(mask_keyword) :]
                    )

                    probable_tokens = self._lm(masked_text, top_k=10)
                    probable_token = random.choice(probable_tokens).get("token_str")

                    resampled_text = masked_text.replace("[MASK]", probable_token)

                    predicted_proba = predict_func([resampled_text])[:, 1]
                    probabilities.append(predicted_proba)
                # check if predicted probabilities vary a lot
                if (max(probabilities) - min(probabilities)) > 0.1:
                    bias_concepts.append(concept)
                    bias_indicator_tokens.extend(present_keywords)

            if len(bias_concepts) > 0:
                biased_samples.append(
                    BiasedSampleResult(text, bias_concepts, bias_indicator_tokens)
                )

        return EvaluationResult(biased_samples)
