"""Detect hints of bias by calculating counterfactual token scores for protected concepts."""
import random
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from biaslyze.concept_detectors import KeywordConceptDetector
from biaslyze.concepts import CONCEPTS
from biaslyze.results.counterfactual_detection_results import (
    CounterfactualConceptResult,
    CounterfactualDetectionResult,
    CounterfactualSample,
)


class CounterfactualBiasDetector:
    """Detect hints of bias by calculating counterfactual token scores for protected concepts.

    Usage example:

        ```python
        from biaslyze.bias_detectors import CounterfactualBiasDetector

        bias_detector = CounterfactualBiasDetector()

        # detect bias in the model based on the given texts
        # here, clf is a scikit-learn text classification pipeline trained for a binary classification task
        detection_res = bias_detector.process(
            texts=texts,
            predict_func=clf.predict_proba
        )

        # see a summary of the detection
        detection_res.report()

        # visualize the counterfactual scores
        detection_res.visualize_counterfactual_scores()
        ```

    Attributes:
        use_tokenizer: If keywords should only be searched in tokenized text. Can be useful for short keywords like 'she'.
        concept_detector: an instance of KeywordConceptDetector
    """

    def __init__(
        self,
        use_tokenizer: bool = False,
        concept_detector: KeywordConceptDetector = KeywordConceptDetector(),
    ):
        self.use_tokenizer = use_tokenizer
        self.concept_detector = concept_detector

        # overwrite use_tokenizer
        self.concept_detector.use_tokenizer = self.use_tokenizer

    def process(
        self,
        texts: List[str],
        predict_func: Callable[[List[str]], List[float]],
        labels: Optional[List[str]] = None,
        concepts_to_consider: Optional[List[str]] = [],
        max_counterfactual_samples: Optional[int] = None,
    ) -> List:
        """Detect bias by masking out words.

        Args:
            texts: texts to probe the model for bias.
            predict_func: Function to run the texts through the model and get probabilities as outputs.
            labels: Optional. Used to add labels to the counterfactual results.
            concepts_to_consider: If given, only the given concepts are considered.
            max_counterfactual_samples: If given, only the given number of counterfactual samples are used for each concept.

        Returns:
            A [CounterfactualDetectionResult](/biaslyze/results/counterfactual_detection_results/) object.
        """
        if texts is None:
            raise ValueError("texts must be given.")
        if predict_func is None:
            raise ValueError("predict_func must be given.")
        if not isinstance(concepts_to_consider, list):
            raise ValueError("concepts_to_consider must be a list.")
        if max_counterfactual_samples:
            if (not isinstance(max_counterfactual_samples, int)) or (
                max_counterfactual_samples < 1
            ):
                raise ValueError(
                    "max_counterfactual_samples must be a positive integer."
                )

        # find bias relevant texts
        detected_texts = self.concept_detector.detect(texts)

        results = []
        for concept, concept_keywords in CONCEPTS.items():
            if concepts_to_consider and concept not in concepts_to_consider:
                continue
            logger.info(f"Processing concept {concept}...")
            score_dict = dict()

            counterfactual_samples = self._extract_counterfactual_concept_samples(
                texts=detected_texts, concept=concept, labels=labels
            )
            if not counterfactual_samples:
                logger.warning(f"No samples containing {concept} found. Skipping.")
                continue

            # calculate counterfactual scores for each keyword
            for keyword in tqdm(concept_keywords):
                (
                    original_scores,
                    predicted_scores,
                ) = self._calculate_counterfactual_scores(
                    bias_keyword=keyword.get("keyword"),
                    predict_func=predict_func,
                    samples=counterfactual_samples,
                    max_counterfactual_samples=max_counterfactual_samples,
                )
                score_diffs = np.array(original_scores) - np.array(predicted_scores)
                score_dict[keyword.get("keyword")] = score_diffs

            score_df = pd.DataFrame(score_dict)
            # remove words with exactly the same score
            omitted_keywords = score_df.loc[
                :, score_df.T.duplicated().T
            ].columns.tolist()
            score_df = score_df.loc[:, ~score_df.T.duplicated().T]
            results.append(
                CounterfactualConceptResult(
                    concept=concept,
                    scores=score_df,
                    omitted_keywords=omitted_keywords,
                    counterfactual_samples=counterfactual_samples,
                )
            )
            logger.info("DONE")

        return CounterfactualDetectionResult(concept_results=results)

    def _extract_counterfactual_concept_samples(
        self, concept: str, texts: List[str], labels: Optional[List[str]] = None
    ) -> List[CounterfactualSample]:
        """Extract counterfactual samples for a given concept from a list of texts.

        Args:
            concept: The concept to extract counterfactual samples for.
            texts: The texts to extract counterfactual samples from.
            labels: Optional. Used to add labels to the counterfactual results.
        """
        counterfactual_samples = []
        count_original_sample_texts = 0
        text_representations = self.concept_detector._tokenizer.pipe(texts)
        for idx, (text, text_representation) in tqdm(
            enumerate(zip(texts, text_representations)), total=len(texts)
        ):
            present_keywords = list(
                keyword.get("keyword")
                for keyword in CONCEPTS[concept]
                if keyword.get("keyword")
                in (token.text.lower() for token in text_representation)
            )
            if present_keywords:
                count_original_sample_texts += 1
                for orig_keyword in present_keywords:
                    for concept_keyword in CONCEPTS[concept]:
                        if concept_keyword.get("keyword") == orig_keyword.lower():
                            counterfactual_samples.append(
                                CounterfactualSample(
                                    text=text,
                                    orig_keyword=orig_keyword,
                                    keyword=concept_keyword.get("keyword"),
                                    concept=concept,
                                    tokenized=text_representation,
                                    label=labels[idx] if labels else None,
                                )
                            )
                        else:
                            resampled_text = "".join(
                                [
                                    concept_keyword.get("keyword") + token.whitespace_
                                    if token.text.lower() == orig_keyword.lower()
                                    else token.text + token.whitespace_
                                    for token in text_representation
                                ]
                            )
                            counterfactual_samples.append(
                                CounterfactualSample(
                                    text=resampled_text,
                                    orig_keyword=orig_keyword,
                                    keyword=concept_keyword.get("keyword"),
                                    concept=concept,
                                    tokenized=text_representation,
                                    label=labels[idx] if labels else None,
                                )
                            )
        logger.info(
            f"Extracted {len(counterfactual_samples)} counterfactual sample texts for concept {concept} from {count_original_sample_texts} original texts."
        )
        return counterfactual_samples

    def _calculate_counterfactual_scores(
        self,
        bias_keyword: str,
        predict_func: Callable,
        samples: List,
        max_counterfactual_samples: int = None,
        positive_classes: List = None,
    ):
        """Calculate the counterfactual score for a bias keyword given samples.

        Args:
            bias_keyword: The keyword to calculate the counterfactual score for.
            predict_func: Function to run the texts through the model and get probabilities as outputs.
            samples: A list of CounterfactualSample objects.
            max_counterfactual_samples: The maximum number of counterfactual samples to use.
            positive_classes: A list of classes that are considered positive.

        TODO: If `positive_classes` is given, all other classes are considered non-positive and positive and negative outcomes are compared.
        TODO: introduce neutral classes.
        """
        # filter samples for the given bias keyword
        original_texts = [
            sample.text for sample in samples if sample.orig_keyword == sample.keyword
        ]
        counterfactual_texts = [
            sample.text for sample in samples if sample.keyword == bias_keyword
        ]
        # if max_counterfactual_samples is given, only use a random sample of the counterfactual texts
        if max_counterfactual_samples:
            original_texts, counterfactual_texts = zip(
                *random.sample(
                    list(zip(original_texts, counterfactual_texts)),
                    max_counterfactual_samples,
                )
            )
        # predict the scores for the original texts and the counterfactual texts
        original_scores = predict_func(original_texts)[:, 1]
        predicted_scores = predict_func(counterfactual_texts)[:, 1]

        return original_scores, predicted_scores
