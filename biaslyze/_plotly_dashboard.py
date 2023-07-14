"""This file contains the new plotting with plotly and dash."""

from collections import defaultdict
import pandas as pd

import plotly.graph_objects as go
from plotly.colors import n_colors
import dash
from dash import dcc
import dash_html_components as html
from dash.dependencies import Input, Output


def _get_default_results(result, concept: str) -> pd.DataFrame:
    dataf = result._get_result_by_concept(concept=concept)
    sort_index = dataf.median().abs().sort_values(ascending=True)
    return dataf[sort_index.index]


def _get_ksr_results(result, concept: str) -> pd.DataFrame:
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


def _build_data_lookup(results):
    concepts = [res.concept for res in results.concept_results]

    lookup = {}
    for method in ["default", "ksr"]:
        method_dict = {}
        for concept in concepts:
            scores = (
                _get_default_results(results, concept=concept)
                if (method == "default")
                else _get_ksr_results(results, concept=concept)
            )
            samples = results._get_counterfactual_samples_by_concept(concept)
            concept_dict = {
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

            method_dict[concept] = concept_dict
        lookup[method] = method_dict
    return lookup


def _plot_dashboard(results, num_keywords: int = 10):
    """Plot a dashboard of the results as interactive boxplots.

    Args:
        results: The results.
        num_keywords: The number of keywords to plot.
    """
    concepts = [res.concept for res in results.concept_results]
    data_lookup = _build_data_lookup(results)

    app = dash.Dash(__name__)

    pink2blue_colormap = n_colors('rgb(0, 152, 218)','rgb(246, 173, 175)', num_keywords, colortype='rgb')


    def generate_box_plot(dataf):
        fig = go.Figure()
        for keyword, color in zip(dataf.columns, pink2blue_colormap):
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
            xaxis_title="Counterfactual Score - difference from zero indicates change of model prediction"
        )
        return fig

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H2("Counterfactual score"),
                    dcc.RadioItems(
                        options=[
                            {"label": "Keyword-based (default)", "value": "default"},
                            {"label": "Sample-based (ksr)", "value": "ksr"},
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
                }
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
        if method in ["default", "ksr"]:
            df = (
                data_lookup[method][concepts[concept_idx]]["data"]
                .iloc[:, -num_keywords:]
                .dropna(how="all")
            )
        else:
            raise ValueError(f"Unknown method '{method}'")
        fig = generate_box_plot(df)
        # Stylize graph
        fig.update_layout(
            template="plotly_white",
            showlegend=False,
            # width=100*4,
            height=50 * num_keywords,
            hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
        )
        return fig

    @app.callback(
        Output("selected-text", "children"),
        Input("box-plot", "clickData"),
        Input("concept-dropdown", "value"),
        Input("select-method", "value"),
    )
    def display_selected_text(click_data, concept_idx, method):
        if click_data is not None:
            keyword = click_data["points"][0]["y"]
            value = click_data["points"][0]["x"]
            index = click_data["points"][0]["pointIndex"]
            try:
                return html.Div(
                    [
                        html.H4("Selected sample:"),
                        html.Table([
                            html.Tr([html.Th("Keyword"), html.Th("Original keyword"), html.Th("Score")]),
                            html.Tr([
                                html.Td(keyword),
                                html.Td(data_lookup[method][concepts[concept_idx]]['original_keyword'][keyword][index]),
                                html.Td(f"{value:.3}"),
                            ])
                        ], style={"padding": "0 0px", "border-collapse": "separate", "border-spacing": "30px 0px"}),
                        html.P(
                            f"{data_lookup[method][concepts[concept_idx]]['texts'][keyword][index]}"
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
        else:
            return ""

    app.run_server(
        mode="inline",
        port=8090,
        dev_tools_ui=True,
        dev_tools_hot_reload=True,
        threaded=True,
    )
