"""Detect bias by counterfactual token fairness methods."""

import numpy as np
import pandas as pd
from tqdm import tqdm
from biaslyze.concepts import CONCEPTS
import matplotlib.pyplot as plt
from typing import List
import spacy


def calculate_counterfactual_sample_score(
    sample: CounterfactualSample, concept: str, clf
):
    # replace the keyword in the sample by all concept keywords and then predict
    original_score = clf.predict_proba([sample.text])[:, 1]
    replaced_texts = []
    for keyword in CONCEPTS[concept]:
        resampled_text = "".join(
            [
                keyword + token.whitespace_
                if token.text.lower() == sample.keyword.lower()
                else token.text + token.whitespace_
                for token in sample.tokenized
            ]
        )
        replaced_texts.append(resampled_text)

    predicted_scores = clf.predict_proba(replaced_texts)[:, 1]
    original_scores = np.ones(predicted_scores.shape) * original_score
    return original_scores, predicted_scores
