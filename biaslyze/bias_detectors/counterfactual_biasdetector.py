"""Detect hints of bias by calculating counterfactual token scores for protected concepts."""
import random
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
import spacy

from biaslyze.concept_detectors import KeywordConceptDetector
from biaslyze.concepts import CONCEPTS
from biaslyze.results.counterfactual_detection_results import (
    CounterfactualConceptResult,
    CounterfactualDetectionResult,
    CounterfactualSample,
)


class CounterfactualBiasDetector:
    """Detect hints of bias by calculating counterfactual token scores for protected concepts.

    The counterfactual score is defined as the difference between the predicted
    probability score for the original text and the predicted probability score for the counterfactual text.

    $$counterfactual_score = P(x=1|counterfactual_text) - P(x=1|original_text),$$

    where counterfactual text is defined as the original text where a keyword of the given concept is
    replaced by another keyword of the same concept. So a counterfactual_score > 0 means that the
    model is more likely to predict the positive class for the original text than for the counterfactual text.

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
        detection_res.visualize_counterfactual_scores(concept="religion")

        # visualize the counterfactual sample scores
        detection_res.visualize_counterfactual_score_by_sample_histogram(concepts=["religion", "gender"])
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
        """Detect potential bias in the model based on the given texts.

        Args:
            texts: texts to probe the model for bias.
            predict_func: Function to run the texts through the model and get probabilities as outputs.
            labels: Optional. Used to add labels to the counterfactual results.
            concepts_to_consider: If given, only the given concepts are considered.
            max_counterfactual_samples: If given, only the given number of counterfactual samples are used for each concept.

        Returns:
            A [CounterfactualDetectionResult](/biaslyze/results/counterfactual_detection_results/) object.

        Raises:
            ValueError: If texts or predict_func is not given.
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

            counterfactual_samples = _extract_counterfactual_concept_samples(
                texts=detected_texts,
                concept=concept,
                tokenizer=self.concept_detector._tokenizer,
                labels=labels,
            )
            if not counterfactual_samples:
                logger.warning(f"No samples containing {concept} found. Skipping.")
                continue

            # calculate counterfactual scores for each keyword
            for keyword in tqdm(concept_keywords):
                # get the counterfactual scores
                counterfactual_scores = _calculate_counterfactual_scores(
                    bias_keyword=keyword.get("keyword"),
                    predict_func=predict_func,
                    samples=counterfactual_samples,
                    max_counterfactual_samples=max_counterfactual_samples,
                )
                # add to score dict
                score_dict[keyword.get("keyword")] = counterfactual_scores
                # add scores to samples
                original_keyword_samples = [
                    sample
                    for sample in counterfactual_samples
                    if (sample.keyword == keyword.get("keyword"))
                    and (sample.keyword == sample.orig_keyword)
                ]
                for score, sample in zip(
                    counterfactual_scores, original_keyword_samples
                ):
                    sample.score = score

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
    concept: str,
    texts: List[str],
    tokenizer: spacy.tokenizer.Tokenizer,
    labels: Optional[List[str]] = None,
) -> List[CounterfactualSample]:
    """Extract counterfactual samples for a given concept from a list of texts.

    A counterfactual sample is defined as a text where a keyword of the
    given concept is replaced by another keyword of the same concept.

    Args:
        concept: The concept to extract counterfactual samples for.
        texts: The texts to extract counterfactual samples from.
        tokenizer: The tokenizer to use for tokenization.
        labels: Optional. Used to add labels to the counterfactual results.
    """
    counterfactual_samples = []
    original_texts = []
    text_representations = tokenizer.pipe(texts)
    concept_keywords = set([keyword.get("keyword") for keyword in CONCEPTS[concept]])
    for idx, (text, text_representation) in tqdm(
        enumerate(zip(texts, text_representations)), total=len(texts)
    ):
        present_keywords = list(
            keyword
            for keyword in concept_keywords
            if keyword in (token.text.lower() for token in text_representation)
        )
        if present_keywords:
            original_texts.append(text)
            for orig_keyword in present_keywords:
                for concept_keyword in concept_keywords:
                    resampled_text = "".join(
                        [
                            concept_keyword + token.whitespace_
                            if token.text.lower() == orig_keyword.lower()
                            else token.text + token.whitespace_
                            for token in text_representation
                        ]
                    )
                    counterfactual_samples.append(
                        CounterfactualSample(
                            text=resampled_text,
                            orig_keyword=orig_keyword,
                            keyword=concept_keyword,
                            concept=concept,
                            tokenized=text_representation,
                            label=labels[idx] if labels else None,
                            source_text=text,
                        )
                    )
    logger.info(
        f"Extracted {len(counterfactual_samples)} counterfactual sample texts for concept {concept} from {len(original_texts)} original texts."
    )
    return counterfactual_samples


def _calculate_counterfactual_scores(
    bias_keyword: str,
    predict_func: Callable,
    samples: List[CounterfactualSample],
    max_counterfactual_samples: int = None,
    positive_classes: Optional[List] = None,
) -> np.ndarray:
    """Calculate the counterfactual score for a bias keyword given samples.

    Args:
        bias_keyword: The keyword to calculate the counterfactual score for.
        predict_func: Function to run the texts through the model and get probabilities as outputs.
        samples: A list of CounterfactualSample objects.
        max_counterfactual_samples: The maximum number of counterfactual samples to use.
        positive_classes: A list of classes that are considered positive.

    TODO: If `positive_classes` is given, all other classes are considered non-positive and positive and negative outcomes are compared.
    TODO: introduce neutral classes.

    Returns:
        A numpy array of differences between the original predictions and the predictions for the counterfactual samples.
        We call this the **counterfactual score**:  counterfactual_score = P(x=1|counterfactual_text) - P(x=1|original_text).

    Raises:
        ValueError: If `positive_classes` is given but the model is not a binary classifier.
        IndexError: If `positive_classes` is given but the model does not have the given classes.
    """
    # filter samples for the given bias keyword
    original_texts = [
        sample.source_text for sample in samples if (sample.keyword == bias_keyword)
    ]
    counterfactual_texts = [
        sample.text for sample in samples if (sample.keyword == bias_keyword)
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
    original_scores = predict_func(original_texts)
    predicted_scores = predict_func(counterfactual_texts)

    # check if the model is a binary classifier
    if (not positive_classes) and (len(original_scores[0]) != 2):
        raise NotImplementedError(
            "Multi-class classification is not yet supported for counterfactual detection."
            "Please use a binary classifier."
            "If you are using a multi-class classifier, please specify the positive classes."
        )

    # calculate score differences
    if positive_classes:
        # sum up the scores for the positive classes and take the difference
        try:
            score_diffs = (
                np.array(predicted_scores[:, positive_classes]).sum(axis=1),
                -np.array(original_scores[:, positive_classes]).sum(axis=1),
            )
        except IndexError:
            raise IndexError(
                f"Positive classes {positive_classes} not found in predictions."
            )
    else:
        score_diffs = np.array(predicted_scores[:, 1]) - np.array(original_scores[:, 1])
    return score_diffs
