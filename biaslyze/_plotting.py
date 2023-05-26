"""This module contains functions for plotting results and metrics."""
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, DataTable, TableColumn
from bokeh.palettes import Spectral
from bokeh.plotting import figure
from bokeh.themes import Theme


def _plot_box_plot(dataf: pd.DataFrame, top_n: Optional[int] = None):
    """Plot a box plot of scores.

    Args:
        dataf: A dataframe with scores for each sample.
        top_n: Only plot the top n concepts.

    Returns:
        A matplotlib axis.
    """

    # sort the dataframe by median absolute value
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
    return ax


def _plot_histogram_dashboard(
    texts: List[str],
    concepts: List[str],
    scores: List[float],
    keywords: List[str],
    keyword_positions: Optional[List[int]] = None,
    num_tokens: Optional[List[int]] = None,
    top_words: Optional[List[str]] = None,
    use_position: bool = False,
    score_version: str = "LimeScore",
):
    """Plot a histogram dashboard of scores.

    Args:
        texts: The texts of the samples.
        concepts: The concepts of the samples.
        scores: The scores of the samples.
        keywords: The keywords of the samples.
        keyword_positions: The positions of the keywords in the samples.
        num_tokens: The number of tokens in the samples.
        top_words: The top words of the samples.
        use_position: If True, use the normalized position for plotting.
        score_version: The name of the score version.
    """

    # dataframe used for plotting
    plot_dataf = pd.DataFrame(
        {
            "text": texts,
            "bias_concepts": concepts,
            "score": scores,
            "bias_keywords": keywords,
        }
    )

    if keyword_positions:
        plot_dataf["keyword_position"] = keyword_positions
    if num_tokens:
        plot_dataf["num_tokens"] = num_tokens
    if top_words:
        plot_dataf["top_words"] = top_words

    if isinstance(concepts[0], list):
        plot_dataf["bias_concepts"] = plot_dataf.bias_concepts.apply(
            lambda x: ", ".join(x)
        )
    if isinstance(keywords[0], list):
        plot_dataf["bias_keywords"] = plot_dataf.bias_keywords.apply(
            lambda x: ", ".join(x)
        )
    # the normalized position might be used instead of score
    if use_position and score_version == "LimeScore":
        plot_dataf["score"] = 1 - (plot_dataf.keyword_position / plot_dataf.num_tokens)
        score_version = "PositionLimeScore"

    def bkapp(doc):
        # update function for selection in histogram
        # define the table columns
        columns = [
            TableColumn(field="text", title="text", width=800),
            TableColumn(field="bias_concepts", title="bias_concepts", width=100),
            TableColumn(field="bias_keywords", title="bias_keywords", width=100),
            TableColumn(field="score", title=score_version, width=50),
        ]

        if keyword_positions:
            columns.append(
                TableColumn(
                    field="keyword_position", title="Keyword Position", width=50
                )
            )
        if num_tokens:
            columns.append(
                TableColumn(field="num_tokens", title="Number of Tokens", width=50)
            )

        source = ColumnDataSource(plot_dataf)
        data_table = DataTable(source=source, columns=columns, width=1200)

        # define histogram part
        n_bins = 30 if use_position else 50
        xmin, xmax = 0 if use_position else min(plot_dataf.score), max(plot_dataf.score)
        xmin = xmin - 0.1 * (xmax - xmin)
        xmax = xmax + 0.1 * (xmax - xmin)
        concepts_present = plot_dataf.bias_concepts.unique().tolist()
        bar_alpha = 0.6
        total_range = np.linspace(xmin, xmax, n_bins)

        # update function for selection in histogram
        def update(attr, old, new):
            """Callback used for plot update when lasso selecting"""
            if new:
                new_min_score, new_max_score = (
                    total_range[new[0]],
                    total_range[new[-1]],
                )
                subset = plot_dataf[
                    (plot_dataf.score > new_min_score)
                    & (plot_dataf.score < new_max_score)
                ].sort_values(by=["score"], ascending=False)
            else:
                subset = plot_dataf.copy()
            source.data = subset

        # plot
        p = figure(
            x_range=(xmin, xmax),
            height=600,
            width=1200,
            title=f"{score_version} of present bias concepts",
            tools="pan,wheel_zoom,xbox_select,reset",
            active_drag="xbox_select",
        )

        for bias_concept, color in zip(concepts_present, Spectral[11]):
            # calculate the histogram
            _hist, _ = np.histogram(
                plot_dataf[plot_dataf.bias_concepts == bias_concept].score,
                bins=n_bins,
                range=(xmin, xmax),
            )
            # create the bar plot for this histogram
            bar = p.vbar(
                x=total_range,
                top=_hist,
                width=(xmax - xmin) / n_bins,
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
        p.legend.click_policy = "hide"
        p.legend.title = "Protected Concept"
        p.legend.title_text_font_style = "bold"
        p.legend.title_text_font_size = "16px"

        # put everything together
        doc.add_root(column(p, data_table))
        doc.theme = Theme(
            json=yaml.load(
                """
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
        """,
                Loader=yaml.FullLoader,
            )
        )

    return bkapp
