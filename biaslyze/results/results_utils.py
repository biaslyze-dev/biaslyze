"""Utility functions for results module."""
from collections import defaultdict
from typing import Any, Dict

import pandas as pd

from .counterfactual_detection_results import CounterfactualDetectionResult


def _get_default_results(
    result: CounterfactualDetectionResult, concept: str
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
    result: CounterfactualDetectionResult, concept: str
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
