"""Detect hints of bias by calculating counterfactual token scores for protected concepts."""
import random
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
        concepts_to_consider: List = None,
        max_counterfactual_samples: int = None,
    ) -> List:
        """Detect bias by masking out words.

        Args:
            texts: texts to probe the model for bias.
            predict_func: Function to run the texts through the model and get probabilities as outputs.
            concepts_to_consider: If given, only the given concepts are considered.
            max_counterfactual_samples: If given, only the given number of counterfactual samples are used for each concept.

        Returns:
            A CounterfactualDetectionResult object.
        """
        # find bias relevant texts
        detected_texts = self.concept_detector.detect(texts)

        results = []
        for concept, concept_keywords in CONCEPTS.items():
            if concepts_to_consider and concept not in concepts_to_consider:
                continue
            logger.info(f"Processing concept {concept}...")
            score_dict = dict()

            counterfactual_samples = self._extract_counterfactual_concept_samples(
                texts=detected_texts, concept=concept
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

    def _extract_counterfactual_concept_samples(self, concept: str, texts: List[str]):
        counterfactual_samples = []
        count_original_sample_texts = 0
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


class CounterfactualSample:
    """A sample for counterfactual bias detection.

    Attributes:
        text: The original text.
        orig_keyword: The original keyword.
        keyword: The keyword that replaced the original keyword.
        concept: The concept that was detected in the text.
        tokenized: The tokenized text in spacy representation.
    """

    def __init__(
        self,
        text: str,
        orig_keyword: str,
        keyword: str,
        concept: str,
        tokenized: List[str],
    ):
        self.text = text
        self.orig_keyword = orig_keyword
        self.keyword = keyword
        self.concept = concept
        self.tokenized = tokenized

    def __repr__(self):
        return f"concept={self.concept}; keyword={self.keyword}; text={self.text}"


class CounterfactualConceptResult:
    """The result of a counterfactual bias detection run for a single concept."""

    def __init__(
        self,
        concept: str,
        scores: pd.DataFrame,
        omitted_keywords: List[str],
        counterfactual_samples: List[CounterfactualSample] = None,
    ):
        self.concept = concept
        self.scores = scores
        self.omitted_keywords = omitted_keywords
        self.counterfactual_samples = counterfactual_samples


class CounterfactualDetectionResult:
    """The result of a counterfactual bias detection run.

    Attributes:
        concept_results: A list of CounterfactualConceptResult objects.
    """

    def __init__(self, concept_results: List[CounterfactualConceptResult]):
        self.concept_results = concept_results

    def _get_result_by_concept(self, concept: str) -> pd.DataFrame:
        for concept_result in self.concept_results:
            if concept_result.concept == concept:
                return concept_result.scores

    def _get_counterfactual_samples_by_concept(
        self, concept: str
    ) -> List[CounterfactualSample]:
        """Get all counterfactual samples for a given concept."""
        for concept_result in self.concept_results:
            if concept_result.concept == concept:
                return concept_result.counterfactual_samples

    def report(self):
        """Show an overview of the results.

        Details:
            For each concept, the maximum mean and maximum standard deviation of the counterfactual scores is shown.
        """
        for concept_result in self.concept_results:
            print(
                f"""Concept: {concept_result.concept}\t\tMax-Mean Counterfactual Score: {np.abs(concept_result.scores.mean()).max():.5f}\t\tMax-Std Counterfactual Score: {concept_result.scores.std().max():.5f}"""
            )

    def visualize_counterfactual_scores(self, concept: str, top_n: int = None):
        """
        Visualize the counterfactual scores for a given concept.

        Args:
            concept: The concept to visualize.
            top_n: If given, only the top n keywords are shown.

        Details:
            The counterfactual scores are shown as boxplots. The median of the scores is indicated by a dashed line.
        """
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

    def visualize_counterfactual_sample_scores(self, concept: str):
        """Visualize the counterfactual scores for each sample for a given concept."""
        import yaml
        from sentence_transformers import SentenceTransformer
        from umap import UMAP
        from bokeh.themes import Theme
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, Slider, HoverTool
        from bokeh.palettes import Category10_10
        from bokeh.layouts import column
        from bokeh.io import show

        dataf = self._get_result_by_concept(concept=concept)
        samples = self._get_counterfactual_samples_by_concept(concept=concept)

        # get the original samples
        original_samples = [
            sample for sample in samples if (sample.keyword == sample.orig_keyword)
        ]

        # get text representations in 2d
        docs = [sample.text for sample in original_samples]
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = sentence_model.encode(docs, show_progress_bar=True)

        # Train BERTopic
        reduced_embeddings = UMAP(
            n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine"
        ).fit_transform(embeddings)

        # build the visualization
        def bkapp(doc):
            hover_tool = HoverTool(
                tooltips=[
                    ("Text", "@text"),
                    ("Keywords", "@keywords"),
                    ("Counterfactual score", "@counterfactual_sample_score"),
                ]
            )
            p = figure(
                width=1200,
                height=800,
                tools=["pan", "wheel_zoom", "box_zoom", "reset", hover_tool],
            )

            bias_intensity = np.array(dataf.mean(axis=1))

            # configure
            df = pd.DataFrame(
                dict(
                    text=[sample.text for sample in original_samples],
                    keywords=[sample.keyword for sample in original_samples],
                    x=reduced_embeddings[:, 0],
                    y=reduced_embeddings[:, 1],
                    color=[
                        Category10_10[idx] for idx in (bias_intensity <= 0).astype(int)
                    ],
                    bias_intensity=500 * np.abs(bias_intensity),
                    counterfactual_sample_score=np.round(bias_intensity, 4),
                )
            )
            source = ColumnDataSource(data=df)

            # add a circle renderer with a size, color, and alpha
            p.scatter(
                "x",
                "y",
                source=source,
                color="color",
                size="bias_intensity",
                alpha=0.3,
            )

            # p.legend.location = "top_left"
            # p.legend.click_policy="hide"

            # slider
            threshold = Slider(
                title="threshold",
                value=0.0,
                start=0.0,
                end=df.bias_intensity.max(),
                step=0.01,
                width=750,
            )

            def update_data(attrname, old, new):
                # Get the current slider values
                t = threshold.value
                new_df = df.copy()
                new_df["bias_intensity"] = new_df.bias_intensity.apply(
                    lambda x: x if x >= t else 0.0
                )
                source.data = new_df

            threshold.on_change("value", update_data)

            doc.add_root(column(threshold, p, width=800))

            # show the results
            doc.theme = Theme(
                json=yaml.load(
                    """
                attrs:
                    figure:
                        background_fill_color: "#DDDDDD"
                        outline_line_color: white
                        toolbar_location: above
                        height: 800
                        width: 1200
                    Grid:
                        grid_line_dash: [6, 4]
                        grid_line_color: white
            """,
                    Loader=yaml.FullLoader,
                )
            )

        show(bkapp)
