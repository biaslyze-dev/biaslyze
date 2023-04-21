"""Contains classes to evaluate the bias of detected concepts."""
import random
import warnings
from typing import List

import numpy as np
import spacy
from eli5.lime import TextExplainer
from loguru import logger
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
        # only use unigrams
        self.explainer.vec.ngram_range = (1, 1)

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
        logger.info(f"Started bias detection on {len(texts)} samples...")
        biased_samples = []
        for text in tqdm(texts):
            # use LIME on the given text sample
            self.explainer.fit(text, predict_func)
            # get the explanation from LIME (linear model coefficients and feature names)
            interpret_sample_dict = {
                np.sign(coef) * np.abs(coef) / sum(np.abs(self.explainer.clf_.coef_[0])): token
                for coef, token in zip(
                    self.explainer.clf_.coef_[0],
                    self.explainer.vec_.get_feature_names_out(),
                )
            }
            # get the most important tokens from the explanation
            top_interpret_sample_dict = sorted(
                interpret_sample_dict.items(), key=lambda x: -np.abs(x[0])
            )[: min(len(interpret_sample_dict), top_n)]
            important_tokens = [w.lower() for (_, w) in top_interpret_sample_dict]
            token_scores = [c for (c, _) in top_interpret_sample_dict]

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
                    BiasedSampleResult(
                        text=text,
                        bias_concepts=bias_concepts,
                        bias_reasons=bias_indicator_tokens,
                        top_words=important_tokens,
                        num_tokens=len(interpret_sample_dict),
                        keyword_position=max([important_tokens.index(bias_token) for bias_token in bias_indicator_tokens]),
                        score=max([token_scores[important_tokens.index(bias_token)] for bias_token in bias_indicator_tokens])
                    )
                )

        return EvaluationResult(biased_samples)


class MaskedBiasEvaluator:
    def __init__(
        self,
    ):
        self._lm = pipeline("fill-mask", model="distilbert-base-uncased")
        self._tokenizer = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])

    def evaluate(
        self, predict_func, texts: List[str], n_resample_keywords: int = 10
    ) -> EvaluationResult:
        """Evaluate if a bias is present with masked language models.

        Use a masked language model to resample keywords in texts and measure the difference in prediction.
        If the difference is 'large' we call is biased sample.

        Note:
            The language model might contain bias itself and resampling might be not diverse.
        """
        logger.info(f"Started bias detection on {len(texts)} samples...")

        biased_samples = []
        for text in tqdm(texts):
            bias_concepts = []
            bias_indicator_tokens = []
            scores = []
            for concept, concept_keywords in CONCEPTS.items():
                # detect if there might be bias
                text_representation = self._tokenizer(text)
                present_keywords = list(
                    keyword
                    for keyword in concept_keywords
                    if keyword in (token.text.lower() for token in text_representation)
                )
                if not present_keywords:
                    continue

                # predict the original sample
                original_score = predict_func([text])[:, 1][0]

                # create the resampled dataset of texts
                resampled_texts = []
                for _ in range(n_resample_keywords):
                    mask_keyword = random.choice(present_keywords)

                    # TODO: Does it make sense to use the language model?
                    # probable_tokens = self._lm(masked_text, top_k=10)
                    # probable_token = random.choice(probable_tokens).get("token_str")

                    # for now we sample from concept keywords
                    probable_token = random.choice(concept_keywords)

                    resampled_text = "".join(
                        [
                            probable_token + token.whitespace_
                            if token.text.lower() == mask_keyword.lower()
                            else token.text + token.whitespace_
                            for token in text_representation
                        ]
                    )
                    resampled_texts.append(resampled_text)

                predicted_scores = predict_func(resampled_texts)[:, 1]

                # check if predicted probabilities vary a lot
                # score = max(predicted_scores) - min(predicted_scores)
                score = np.max(np.abs(predicted_scores - original_score))
                if score > 0.0:
                    bias_concepts.append(concept)
                    bias_indicator_tokens.extend(present_keywords)
                    scores.append(score)

            if len(bias_concepts) > 0:
                biased_samples.append(
                    BiasedSampleResult(
                        text=text,
                        bias_concepts=bias_concepts,
                        bias_reasons=bias_indicator_tokens,
                        num_tokens=len(set(text_representation)),
                        top_words=[],
                        score=max(scores),
                        keyword_position=0,
                    )
                )

        return EvaluationResult(biased_samples)
