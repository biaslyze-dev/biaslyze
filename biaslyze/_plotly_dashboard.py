"""This file contains the new plotting with plotly and dash."""
from typing import List, Tuple, Dict, Any
from collections import defaultdict

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output
from plotly.colors import n_colors


def _get_default_results(
    result, concept: str
) -> pd.DataFrame:
    """Get the counterfactual scores (default) for each original sample.

    Args:
        result: An instance of CounterfactualDetectionResults.
        concept: The concept to get the results for.

    Returns:
        A DataFrame with the counterfactual scores for each original sample.
    """
    dataf = result._get_result_by_concept(concept=concept)
    sort_index = dataf.median().abs().sort_values(ascending=True)
    return dataf[sort_index.index]


def _get_ksr_results(
    result, concept: str
) -> pd.DataFrame:
    """Get the counterfactual scores (ksr) for each original sample.

    Args:
        result: An instance of CounterfactualDetectionResults.
        concept: The concept to get the results for.

    Returns:
        A DataFrame with the counterfactual scores for each original sample.
    """
    dataf = result._get_result_by_concept(concept=concept)
    samples = result._get_counterfactual_samples_by_concept(concept=concept)

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
    sort_index = counterfactual_df.median().abs().sort_values(ascending=True)
    return counterfactual_df[sort_index.index]


def _build_data_lookup(results) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Build a lookup dictionary for the data.

    Example output:
    {
        "default": {
            "concept1": {
                "data": pd.DataFrame,
                "texts": {
                    "keyword1": [str, str, ...],
                    "keyword2": [str, str, ...],
                    ...
                },
                "original_keyword": {
                    "keyword1": [str, str, ...],
                    "keyword2": [str, str, ...],
                    ...
                }
            },
            "concept2": {
                "data": pd.DataFrame,
                "texts": {
                    "keyword1": [str, str, ...],
                    "keyword2": [str, str, ...],
                    ...
                },
                "original_keyword": {
                    "keyword1": [str, str, ...],
                    "keyword2": [str, str, ...],
                    ...
                }
            },
            ...
        },
        "ksr": {
            "concept1": {
                "data": pd.DataFrame,
                "texts": {
                    "keyword1": [str, str, ...],
                    "keyword2": [str, str, ...],
                    ...
                },
                "original_keyword": {
                    "keyword1": [str, str, ...],
                    "keyword2": [str, str, ...],
                    ...
                }
            },
            "concept2": {
                "data": pd.DataFrame,
                "texts": {
                    "keyword1": [str, str, ...],
                    "keyword2": [str, str, ...],
                    ...
                },
                "original_keyword": {
                    "keyword1": [str, str, ...],
                    "keyword2": [str, str, ...],
                    ...
                }
            },
            ...
        },
        "histogram": {
            ...
        }
    }

    Args:
        results: An instance of CounterfactualDetectionResults.
    """
    concepts = [res.concept for res in results.concept_results]

    lookup = {}
    for method in ["default", "ksr", "histogram"]:
        method_dict = {}
        for concept in concepts:
            scores = (
                _get_default_results(results, concept=concept)
                if (method == "default")
                else _get_ksr_results(results, concept=concept)
            )
            samples = results._get_counterfactual_samples_by_concept(concept)
            method_dict[concept] = {
                "data": scores,
                "texts": {
                    keyword: [
                        sample.text for sample in samples if sample.keyword == keyword
                    ]
                    for keyword in scores.columns
                },
                "original_keyword": {
                    keyword: [
                        sample.orig_keyword
                        for sample in samples
                        if sample.keyword == keyword
                    ]
                    for keyword in scores.columns
                },
            }

        lookup[method] = method_dict
    return lookup


def _prepare_histogram_display_data(
    range_start: int, range_end: int, concept_data: pd.DataFrame, num_keywords: int
) -> List[Tuple[str, str, float, str]]:
    """Prepare the data for the histogram display.

    Args:
        range_start: The start of the range to filter the data.
        range_end: The end of the range to filter the data.
        concept_data: The data for the concept to filter.
        num_keywords: The number of keywords to display.

    Returns:
        A list of tuples of the form (keyword, original_keyword, score, text).

    Raises:
        ValueError: If the range is invalid.
    """
    if range_start >= range_end:
        raise ValueError("range_start must be less than range_end.")

    df = concept_data["data"].iloc[:, -num_keywords:].dropna(how="all")
    selected = []
    for keyword in df.columns:
        indices = np.where((df[keyword] >= range_start) & (df[keyword] <= range_end))[0]
        selected_texts = [
            concept_data["texts"][keyword][index]
            for index in indices
            if index < len(concept_data["texts"][keyword])
        ]
        selected_original = [
            concept_data["original_keyword"][keyword][index]
            for index in indices
            if index < len(concept_data["original_keyword"][keyword])
        ]
        selected_score = [
            concept_data["data"][keyword][index]
            for index in indices
            if index < len(concept_data["data"][keyword])
        ]

        selected_data = [
            (
                "..."
                + text[
                    max(text.lower().index(keyword) - 50, 0) : text.lower().index(
                        keyword
                    )
                    + len(keyword)
                    + 50
                ],
                original,
                keyword,
                f"{score:.3}",
            )
            for text, original, score in zip(
                selected_texts, selected_original, selected_score
            )
        ]
        selected.extend(selected_data)
    return selected


def _generate_box_plot(dataf: pd.DataFrame, colormap: List[str]) -> go.Figure:
    """Generate a box plot of the data.

    Args:
        dataf: The data to plot.
        colormap: The colormap to use.

    Returns:
        A plotly figure.
    """
    fig = go.Figure()
    for keyword, color in zip(dataf.columns, colormap):
        hover_text = [f"Value: {value:.3}" for value in dataf[keyword]]
        fig.add_trace(
            go.Box(
                x=dataf[keyword],
                orientation="h",
                name=keyword,
                hovertext=hover_text,
                marker=dict(color=color),
            )
        )
    fig.update_layout(
        showlegend=False,
        xaxis_title="Counterfactual Score - difference from zero indicates change of model prediction",
    )
    return fig


def _generate_histogram(dataf: pd.DataFrame, colormap: List[str]) -> go.Figure:
    """Generate a histogram of the data.

    Args:
        dataf: The data to plot.
        colormap: The colormap to use.

    Returns:
        A plotly figure.
    """
    fig = go.Figure()
    plot_data = []
    for keyword in dataf.columns:
        plot_data.extend(dataf[keyword].tolist())

    fig.add_trace(
        go.Histogram(
            x=plot_data,
            marker=dict(color=colormap[3]),
            nbinsx=100,
        )
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="Counterfactual Score",
        yaxis_title="Frequency",
    )
    return fig


def _plot_dashboard(results, num_keywords: int = 10, port: int = 8090):
    """Plot a dashboard of the results as interactive boxplots.

    Args:
        results: The results.
        num_keywords: The number of keywords to plot.
        port: The port to run the dashboard on.
    """
    concepts = [res.concept for res in results.concept_results]
    data_lookup = _build_data_lookup(results)

    app = dash.Dash(__name__)

    # prepare the colormap to use for the plots
    pink2blue_colormap = n_colors(
        "rgb(0, 152, 218)", "rgb(246, 173, 175)", num_keywords, colortype="rgb"
    )

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H2("Counterfactual score"),
                    dcc.RadioItems(
                        options=[
                            {"label": "Keyword-based (default)", "value": "default"},
                            {"label": "Sample-based (ksr)", "value": "ksr"},
                            {"label": "Sample-based (histogram)", "value": "histogram"},
                        ],
                        value="default",
                        inline=True,
                        id="select-method",
                    ),
                    html.H4("Select concept:"),
                    dcc.Dropdown(
                        options=[
                            {"label": concept, "value": idx}
                            for idx, concept in enumerate(concepts)
                        ],
                        value=0,
                        id="concept-dropdown",
                        style={"color": "black"},
                    ),
                ],
                style={
                    "color": "white",
                    "background-color": "#3c9fca",
                    "padding": "10px",
                    "border-radius": "5px",
                    "font-family": "Arial, sans-serif",
                },
            ),
            dcc.Graph(id="box-plot"),
            html.Div(id="selected-text"),
        ]
    )

    @app.callback(
        Output("box-plot", "figure"),
        Input("concept-dropdown", "value"),
        Input("select-method", "value"),
        prevent_initial_callback=True,
    )
    def update_box_plot(concept_idx, method):
        """Update the box plot."""
        if method in ["default", "ksr", "histogram"]:
            df = (
                data_lookup[method][concepts[concept_idx]]["data"]
                .iloc[:, -num_keywords:]
                .dropna(how="all")
            )
        else:
            raise ValueError(f"Unknown method '{method}'")

        if method in ["default", "ksr"]:
            fig = _generate_box_plot(df, colormap=pink2blue_colormap)
        elif method == "histogram":
            fig = _generate_histogram(df, colormap=pink2blue_colormap)

        # Stylize graph
        fig.update_layout(
            template="plotly_white",
            showlegend=False,
            height=45 * num_keywords,
            hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
        )
        return fig

    @app.callback(
        Output("selected-text", "children"),
        Input("box-plot", "clickData"),
        Input("box-plot", "relayoutData"),
        Input("concept-dropdown", "value"),
        Input("select-method", "value"),
    )
    def display_selected_text(click_data, relayout_data, concept_idx, method):
        """Display selected text for the respective plots.

        For plot method `histogram` this resolves to creating a data table with all texts in the selected window.
        For plot method `default` and `ksr` this resolves to displaying the text of the selected sample in the boxplot.
        """
        display_data = data_lookup[method][concepts[concept_idx]]
        if method == "histogram":
            if relayout_data is not None and "xaxis.range[0]" in relayout_data:
                range_start = relayout_data["xaxis.range[0]"]
                range_end = relayout_data["xaxis.range[1]"]
                selected = _prepare_histogram_display_data(
                    range_start=range_start,
                    range_end=range_end,
                    data=display_data,
                    num_keywords=num_keywords,
                )
                if selected:
                    return dash_table.DataTable(
                        data=selected,
                        columns=[
                            {"id": idx, "name": name}
                            for idx, name in enumerate(
                                ["text", "original_keyword", "keyword", "score"]
                            )
                        ],
                        page_size=10,
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                    )
        else:
            if click_data is not None:
                keyword = click_data["points"][0]["y"]
                value = click_data["points"][0]["x"]
                index = click_data["points"][0]["pointIndex"]
                try:
                    raw_text = display_data["texts"][keyword][index]
                    text_with_highlights = (
                        raw_text[: raw_text.lower().index(keyword)]
                        + "<b>"
                        + raw_text[
                            raw_text.lower()
                            .index(keyword) : raw_text.lower()
                            .index(keyword)
                            + len(keyword)
                        ]
                        + "</b>"
                        + raw_text[raw_text.lower().index(keyword) + len(keyword) :]
                    )
                    return html.Div(
                        [
                            html.H4("Selected sample:"),
                            html.Table(
                                [
                                    html.Tr(
                                        [
                                            html.Th("Keyword"),
                                            html.Th("Original keyword"),
                                            html.Th("Score"),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td(keyword),
                                            html.Td(
                                                display_data["original_keyword"][
                                                    keyword
                                                ][index]
                                            ),
                                            html.Td(f"{value:.3}"),
                                        ]
                                    ),
                                ],
                                style={
                                    "padding": "0 0px",
                                    "border-collapse": "separate",
                                    "border-spacing": "30px 0px",
                                },
                            ),
                            dcc.Markdown(
                                f"{text_with_highlights}",
                                dangerously_allow_html=True,
                            ),
                        ],
                        style={
                            "color": "white",
                            "background-color": "#3c9fca",
                            "padding": "10px",
                            "border-radius": "5px",
                            "font-family": "Arial, sans-serif",
                        },
                    )
                except KeyError:
                    return ""
                except IndexError:
                    return ""
        return ""

    app.run_server(
        mode="inline",
        port=port,
        dev_tools_ui=True,
        dev_tools_hot_reload=True,
        threaded=True,
    )
