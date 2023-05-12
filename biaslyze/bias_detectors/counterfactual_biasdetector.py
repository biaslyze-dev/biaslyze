"""Detect hints of bias by calculating counterfactual token scores for protected concepts."""
import numpy as np
import pandas as pd
from typing import Callable, List
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt

from biaslyze.concepts import CONCEPTS
from biaslyze.concept_detectors import KeywordConceptDetector


class CounterfactualBiasDetector:
    """Detect hints of bias by calculating counterfactual token scores for protected concepts.

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
        self, texts: List[str], predict_func: Callable[[List[str]], List[float]]
    ) -> List:
        """Detect bias by masking out words.

        Args:
            texts: texts to probe the model for bias.
            predict_func: Function to run the texts through the model and get probabilities as outputs.
        """
        # find bias relevant texts
        detected_texts = self.concept_detector.detect(texts)

        results = []
        for concept, concept_keywords in CONCEPTS.items():
            logger.info(f"Processing concept {concept}...")
            score_dict = dict()

            samples = self._extract_concept_samples(
                texts=detected_texts, concept=concept
            )
            if not samples:
                logger.warning(f"No samples containing {concept} found. Skipping.")
                continue

            for keyword in tqdm(concept_keywords):
                (
                    original_scores,
                    predicted_scores,
                ) = self._calculate_counterfactual_scores(
                    bias_keyword=keyword.get("keyword"),
                    predict_func=predict_func,
                    samples=samples,
                )
                score_diffs = np.array(original_scores) - np.array(predicted_scores)
                score_dict[keyword.get("keyword")] = score_diffs

            score_df = pd.DataFrame(score_dict)
            # remove words with exactly the same score
            score_df = score_df.loc[:, ~score_df.T.duplicated().T]
            results.append(
                CounterfactualConceptResult(concept=concept, scores=score_df)
            )
            logger.info("DONE")

        return CounterfactualDetectionResult(concept_results=results)

    def _extract_concept_samples(self, concept: str, texts: List[str]):
        samples = []

        text_representations = self.concept_detector._tokenizer.pipe(texts)
        for text, text_representation in tqdm(
            zip(texts, text_representations), total=len(texts)
        ):
            present_keywords = list(
                keyword.get("keyword")
                for keyword in CONCEPTS[concept]
                if keyword.get("keyword")
                in (token.text.lower() for token in text_representation)
            )
            if present_keywords:
                for keyword in present_keywords:
                    samples.append(
                        CounterfactualSample(
                            text=text,
                            keyword=keyword,
                            concept=concept,
                            tokenized=text_representation,
                        )
                    )
        logger.info(f"Extracted {len(samples)} sample texts for concept {concept}")
        return samples

    def _calculate_counterfactual_scores(
        self,
        bias_keyword: str,
        predict_func: Callable,
        samples: List,
        positive_classes: List = None,
    ):
        """Calculate the counterfactual score for a bias keyword given samples.

        TODO: If `positive_classes` is given, all other classes are considered non-positive and positive and negative outcomes are compared.
        TODO: introduce neutral classes.
        """
        # change the text for all of them and predict
        original_scores = predict_func([sample.text for sample in samples])[:, 1]
        replaced_texts = []
        # text_representations = bias_eval._tokenizer.pipe([sample.text for sample in samples])
        for sample in samples:
            resampled_text = "".join(
                [
                    bias_keyword + token.whitespace_
                    if token.text.lower() == sample.keyword.lower()
                    else token.text + token.whitespace_
                    for token in sample.tokenized
                ]
            )
            replaced_texts.append(resampled_text)

        predicted_scores = predict_func(replaced_texts)[:, 1]

        return original_scores, predicted_scores


class CounterfactualSample:
    def __init__(self, text: str, keyword: str, concept: str, tokenized: List[str]):
        self.text = text
        self.keyword = keyword
        self.concept = concept
        self.tokenized = tokenized

    def __repr__(self):
        return f"concept={self.concept}; keyword={self.keyword}; text={self.text}"


class CounterfactualConceptResult:
    def __init__(self, concept: str, scores: pd.DataFrame):
        self.concept = concept
        self.scores = scores


class CounterfactualDetectionResult:
    def __init__(self, concept_results: List[CounterfactualConceptResult]):
        self.concept_results = concept_results

    def _get_result_by_concept(self, concept: str) -> pd.DataFrame:
        for concept_result in self.concept_results:
            if concept_result.concept == concept:
                return concept_result.scores

    def report(self):
        """Show an overview of the results."""
        for concept_result in self.concept_results:
            print(
                f"Concept: {concept_result.concept}\t\tMax-Mean Counterfactual Score: {np.abs(concept_result.scores.mean()).max()}"
            )

    def visualize_counterfactual_scores(self, concept: str, top_n: int = None):
        """"""
        dataf = self._get_result_by_concept(concept=concept)
        sort_index = dataf.median().abs().sort_values(ascending=True)
        sorted_dataf = dataf[sort_index.index]
        if top_n:
            sorted_dataf = sorted_dataf.iloc[:, -top_n:]
        ax = sorted_dataf.plot.box(
            vert=False, figsize=(12, int(sorted_dataf.shape[1] / 2.2))
        )
        ax.vlines(
            x=0,
            ymin=0.5,
            ymax=sorted_dataf.shape[1] + 0.5,
            colors="black",
            linestyles="dashed",
            alpha=0.5,
        )
        ax.set_title(f"Distribution of counterfactual scores for concept '{concept}'")
        ax.set_xlabel(
            "Counterfactual scores - differences from zero indicate the direction of bias."
        )
        plt.show()
