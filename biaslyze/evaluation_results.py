"""Classes to return results of the different steps."""
from collections import Counter, defaultdict
from pprint import pprint
from typing import List
import numpy as np
import pandas as pd
import yaml
from bokeh.layouts import column
from bokeh.models import (ColumnDataSource,
                          DataTable, TableColumn)
from bokeh.plotting import figure
from bokeh.themes import Theme
from bokeh.palettes import Spectral


class BiasedSampleResult:
    """A sample on which the model might behave biased.

    Contains details on why it might be biased and what concepts are affected.

    Attributes:
        text: Sample text
        bias_concepts: Protected concepts present in the text by which the model appears biased.
        bias_reasons: Reasons why bias was detected. Might be a list of keywords.
    """

    def __init__(self, text: str, bias_concepts: List[str], bias_reasons: List[str], top_words: List[str], num_tokens: List[str], keyword_position: int, score: float):
        self.text = text
        self.bias_concepts = bias_concepts
        self.bias_reasons = bias_reasons
        self.top_words = top_words
        self.num_tokens = num_tokens
        self.keyword_position = keyword_position
        self.score = score

    def __repr__(self) -> str:
        return f"''{self.text}'' might contain bias {self.bias_concepts}; reasons: {self.bias_reasons}"


class EvaluationResult:
    """Contains all samples on detected potential bias issues.

    Attribues:
        biased_samples: A list of BiasedSampleResults.
    """

    def __init__(self, biased_samples: List[BiasedSampleResult]):
        self.biased_samples = biased_samples

    def summary(self):
        """Print some summary statistics on the detected bias.

        Answers the following questions:

        - How many samples contain bias for the model?
        - What concepts are affected?
        - What are the reasons for the bias detection?

        Example output:
        ```
        Detected 1 samples with potential issues.
            Potentially problematic concepts detected: Counter({'nationality': 1})
            Based on keywords: Counter({'german': 1}).
        ```
        """
        print(self.__repr__())

    def details(self, group_by_concept: bool = False):
        """Print the details of every biased sample detected.

        Args:
            group_by_concept: If the output should be grouped by concepts.
        """
        if group_by_concept:
            concept_groups = defaultdict(list)
            for sample in self.biased_samples:
                for bias_concept in sample.bias_concepts:
                    concept_groups[bias_concept].append(
                        {"text": sample.text, "reason": sample.bias_reasons}
                    )
            for concept, group in concept_groups.items():
                print(f"Concept: {concept}")
                pprint(group)
        else:
            for sample in self.biased_samples:
                print(sample)

    def __repr__(self) -> str:
        concepts = []
        reasons = []
        for sample in self.biased_samples:
            concepts.extend(sample.bias_concepts)
            reasons.extend(sample.bias_reasons)

        concepts_stats = Counter()
        concepts_stats.update(concepts)
        reasons_stats = Counter()
        reasons_stats.update(reasons)
        representation_string = f"""Detected {len(self.biased_samples)} samples with potential issues.
    Potentially problematic concepts detected: {concepts_stats.most_common(10)}
    Based on keywords: {reasons_stats.most_common(20)}."""
        return representation_string

    def dashboard(self, use_position=False):
        """Return a bokeh dashboard.
        
        Args:
            use_position: If True, use the normalized position for plotting.
        """
        
        res_df = pd.DataFrame({
            "text": [sample.text for sample in self.biased_samples],
            "bias_concepts": [sample.bias_concepts for sample in self.biased_samples],
            "keyword_position": [sample.keyword_position for sample in self.biased_samples],
            "bias_keywords": [sample.bias_reasons for sample in self.biased_samples],
            "num_tokens": [sample.num_tokens for sample in self.biased_samples],
            "top_words": [sample.top_words for sample in self.biased_samples],
            "score": [sample.score for sample in self.biased_samples],
        })

        res_df["bias_concepts_joined"] = res_df.bias_concepts.apply(lambda x: ", ".join(x))
        res_df["bias_keywords_joined"] = res_df.bias_keywords.apply(lambda x: ", ".join(x))
        # the normalized position might be used instead of score
        if use_position:
            res_df["score"] = 1 - (res_df.keyword_position / res_df.num_tokens)

        def bkapp(doc):
            # update function for selection in histogram
            # define the table columns
            columns = [
                TableColumn(field="text", title="text", width=800),
                TableColumn(field="bias_concepts_joined", title="bias_concepts", width=100),
                TableColumn(field="bias_keywords_joined", title="bias_keywords", width=100),
                TableColumn(field="score", title="LimeScore", width=50),
                TableColumn(field="keyword_position", title="Keyword Position", width=50),
                TableColumn(field="num_tokens", title="Num unique tokens", width=50),
            ]
            source = ColumnDataSource(res_df)
            data_table = DataTable(
                source=source, columns=columns, width=1200
            )
            
            # define histogram part
            n_bins = 30 if use_position else 50
            xmin, xmax = 0 if use_position else min(res_df.score), max(res_df.score)
            concepts_present = res_df.bias_concepts_joined.unique().tolist()
            bar_alpha = 0.6
            total_range = np.linspace(xmin, xmax, n_bins)
            
            # update function for selection in histogram
            def update(attr, old, new):
                """Callback used for plot update when lasso selecting"""
                if new:
                    new_min_score, new_max_score = total_range[new[0]], total_range[new[-1]]
                    subset = (
                        res_df[
                            (res_df.score > new_min_score) & (res_df.score < new_max_score)
                        ]
                        .sort_values(by=['score'], ascending=False)
                    )
                else:
                    subset = res_df.copy()
                source.data = subset
            
            # plot
            p = figure(
                x_range=(xmin, xmax),
                height=600,
                width=1200,
                title="LimeScore of present bias concepts",
                tools="pan,wheel_zoom,xbox_select,reset", active_drag="xbox_select"
            )
            
            for bias_concept, color in zip(concepts_present, Spectral[11]):
                # calculate the histogram
                _hist, _ = np.histogram(
                    res_df[res_df.bias_concepts_joined == bias_concept].score,
                    bins=n_bins,
                    range=(xmin, xmax)
                )
                # create the bar plot for this histogram
                bar = p.vbar(
                    x=total_range,
                    top=_hist,
                    width=(xmax - xmin)/n_bins,
                    color=color,
                    alpha=bar_alpha,
                    legend_label=bias_concept,
                )
                # selection handler
                bar.data_source.selected.on_change("indices", update)

            # configure the appearance and the legend
            p.y_range.start = 0
            p.xgrid.grid_line_color = None
            p.legend.orientation = "vertical"
            p.legend.location = "top_right"
            p.legend.click_policy="hide"
            p.legend.title = 'Protected Concept'
            p.legend.title_text_font_style = "bold"
            p.legend.title_text_font_size = "16px"
            
            # put everything together
            doc.add_root(column(p, data_table))
            doc.theme = Theme(json=yaml.load("""
                attrs:
                    figure:
                        background_fill_color: "#DDDDDD"
                        outline_line_color: white
                        toolbar_location: above
                        height: 500
                        width: 800
                    Grid:
                        grid_line_dash: [6, 4]
                        grid_line_color: white
            """, Loader=yaml.FullLoader))
        return bkapp