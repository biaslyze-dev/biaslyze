"""This module contains classes to store and process the results of counterfactual bias detection runs."""
from collections import defaultdict
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from biaslyze._plotting import _plot_box_plot, _plot_histogram_dashboard


class CounterfactualSample:
    """A sample for counterfactual bias detection.

    Attributes:
        text: The original text.
        orig_keyword: The original keyword.
        keyword: The keyword that replaced the original keyword.
        concept: The concept that was detected in the text.
        tokenized: The tokenized text in spacy representation.
        label: The label of the original text.
        source_text: The source text from which the text was derived.
    """

    def __init__(
        self,
        text: str,
        orig_keyword: str,
        keyword: str,
        concept: str,
        tokenized: List[str],
        label: int = None,
        source_text: str = None,
    ):
        self.text = text
        self.orig_keyword = orig_keyword
        self.keyword = keyword
        self.concept = concept
        self.tokenized = tokenized
        self.label = label
        self.source_text = source_text

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
        """Get the result for a given concept.

        Args:
            concept: The concept for which to get the result.

        Returns:
            The DataFrame with results for the given concept.

        Raises:
            ValueError: If the concept is not found in the results.
        """
        for concept_result in self.concept_results:
            if concept_result.concept == concept:
                return concept_result.scores.copy()
        raise ValueError(f"Concept {concept} not found in results.")

    def _get_counterfactual_samples_by_concept(
        self, concept: str
    ) -> List[CounterfactualSample]:
        """Get all counterfactual samples for a given concept.

        Args:
            concept: The concept for which to get the counterfactual samples.

        Returns:
            A list of CounterfactualSample objects.

        Raises:
            ValueError: If the concept is not found in the results.
        """
        for concept_result in self.concept_results:
            if concept_result.concept == concept:
                return concept_result.counterfactual_samples
        raise ValueError(f"Concept {concept} not found in results.")

    def report(self):
        """Show an overview of the results.

        Details:
            For each concept, the maximum mean and maximum standard deviation of the counterfactual scores is shown.
        """
        for concept_result in self.concept_results:
            print(
                f"""Concept: {concept_result.concept}\t\tMax-Mean Counterfactual Score: {np.abs(concept_result.scores.mean()).max():.5f}\t\tMax-Std Counterfactual Score: {concept_result.scores.std().max():.5f}"""
            )

    def visualize_counterfactual_scores(
        self, concept: str, top_n: Optional[int] = None
    ):
        """
        Visualize the counterfactual scores for a given concept.

        The score is calculated by comparing the prediction of the original sample with the prediction of the counterfactual sample.
        For every keyword shown all concept keywords are replaced with the keyword shown. The difference in the prediction to the original is then calculated.

        The counterfactual scores are shown as boxplots. The median of the scores is indicated by a dashed line.

        Args:
            concept: The concept to visualize.
            top_n: If given, only the top n keywords are shown.

        Returns:
            The matplotlib plot.

        Raises:
            ValueError: If the concept is not found in the results.
        """
        dataf = self._get_result_by_concept(concept=concept)
        ax = _plot_box_plot(dataf, top_n=top_n)
        ax.set_title(
            f"Distribution of counterfactual scores for concept '{concept}'\nsorted by median score"
        )
        ax.set_xlabel(
            "Counterfactual scores - differences from zero indicate the direction of bias."
        )
        plt.show()

    def visualize_counterfactual_sample_scores(
        self, concept: str, top_n: Optional[int] = None
    ):
        """Visualize the counterfactual scores given concept.

        This differs from visualize_counterfactual_score_by_sample in that it shows the counterfactual
        score grouped by the original keyword in the text, not the counterfactual keyword.

        Args:
            concept: The concept to visualize.
            top_n: If given, only the top n keywords are shown.
        """
        dataf = self._get_result_by_concept(concept=concept)
        samples = self._get_counterfactual_samples_by_concept(concept=concept)

        # get the original samples
        original_samples = [
            sample for sample in samples if (sample.keyword == sample.orig_keyword)
        ]

        # get the counterfactual scores for each original sample
        counterfactual_plot_dict = defaultdict(list)
        for sample, (_, score) in zip(original_samples, dataf.iterrows()):
            counterfactual_plot_dict[sample.orig_keyword].extend(score.tolist())

        counterfactual_df = pd.DataFrame(
            dict([(k, pd.Series(v)) for k, v in counterfactual_plot_dict.items()])
        )

        # plot
        ax = _plot_box_plot(counterfactual_df, top_n=top_n)
        ax.set_title(
            f"Distribution of counterfactual scores for concept '{concept}' by original keyword\nsorted by median score"
        )
        ax.set_xlabel(
            "Counterfactual scores - differences from zero indicate the direction of bias."
        )
        plt.show()

    def visualize_counterfactual_score_by_sample_histogram(
        self, concepts: Optional[List[str]] = None
    ):
        """Visualize the counterfactual scores for each sample as a histogram.

        Args:
            concepts: If given, only the concepts in this list are shown. Otherwise, all concepts are shown.

        Raises:
            ValueError: If no samples are found for the given concepts.
        """
        all_scores = []
        all_samples = []
        for concept_result in self.concept_results:
            # check if the concept is in the list of concepts to show
            if concepts and (concept_result.concept not in concepts):
                continue
            dataf = concept_result.scores.copy()
            samples = concept_result.counterfactual_samples

            # get the original samples
            original_samples = [
                sample for sample in samples if (sample.keyword == sample.orig_keyword)
            ]

            # get the counterfactual scores for each original sample
            for sample, (_, score) in zip(original_samples, dataf.iterrows()):
                all_samples.append(sample)
                # calculate the median score and change the sign
                # this means that the score represents the median change in prediction if the keyword is replaced
                # with a concept keyword.
                all_scores.append(-1 * np.median(score.tolist()))
        if all_samples == []:
            raise ValueError(
                f"No results found. Please make sure that the concepts are in the results."
            )

        dashboard = _plot_histogram_dashboard(
            texts=[sample.text for sample in all_samples],
            concepts=[sample.concept for sample in all_samples],
            scores=all_scores,
            keywords=[sample.keyword for sample in all_samples],
            score_version="CounterfactualSampleScore",
        )

        return dashboard

    def visualize_counterfactual_score_by_sample(self, concept: str):
        """Visualize the counterfactual scores for each sample for a given concept."""
        import yaml
        from bokeh.layouts import column
        from bokeh.models import ColumnDataSource, HoverTool, Slider
        from bokeh.palettes import Category10_10
        from bokeh.plotting import figure
        from bokeh.themes import Theme
        from sentence_transformers import SentenceTransformer
        from umap import UMAP

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

        return bkapp
