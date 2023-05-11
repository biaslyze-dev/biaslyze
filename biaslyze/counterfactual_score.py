"""Detect bias by counterfactual token fairness methods."""

import numpy as np
import pandas as pd
from tqdm import tqdm
from biaslyze.concepts import CONCEPTS
import matplotlib.pyplot as plt
from typing import List
import spacy


class Sample:
    def __init__(self, text: str, keyword: str, concept: str, tokenized: List[str]):
        self.text = text
        self.keyword = keyword
        self.concept = concept
        self.tokenized = tokenized

    def __repr__(self):
        return f"concept={self.concept}; keyword={self.keyword}; text={self.text}"


def extract_concept_samples(concept: str, texts: List[str], N: int = 1000):
    samples = []
    _tokenizer = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])

    text_representations = _tokenizer.pipe(texts[:N])
    for text, text_representation in tqdm(zip(texts[:N], text_representations)):
        present_keywords = list(
            keyword.get("keyword")
            for keyword in CONCEPTS[concept]
            if keyword.get("keyword") in (token.text.lower() for token in text_representation)
        )
        if present_keywords:
            for keyword in present_keywords:
                samples.append(
                    Sample(
                        text=text,
                        keyword=keyword,
                        concept=concept,
                        tokenized=text_representation,
                    )
                )
    print(f"Extracted {len(samples)} sample texts for concept {concept}")
    return samples


def calculate_counterfactual_score(bias_keyword: str, clf, samples: List[Sample], positive_classes: List = None):
    """Calculate the counterfactual score for a bias keyword given samples.
    
    TODO: If `positive_classes` is given, all other classes are considered non-positive and positive and negative outcomes are compared.
    TODO: introduce neutral classes.
    """
    # change the text for all of them and predict
    original_scores = clf.predict_proba([sample.text for sample in samples])[:, 1]
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

    predicted_scores = clf.predict_proba(replaced_texts)[:, 1]

    # print(f"SenseScore: {np.mean(np.array(original_scores) - np.array(predicted_scores)):.5}")
    return original_scores, predicted_scores


def calculate_all_scores(texts: List[str], concept: str, clf, n_samples=1000):
    score_dict = dict()

    if not n_samples:
        n_samples = len(texts)

    samples = extract_concept_samples(texts=texts, concept=concept, N=n_samples)

    for keyword in tqdm(CONCEPTS[concept]):
        original_scores, predicted_scores = calculate_counterfactual_score(
            bias_keyword=keyword.get("keyword"), clf=clf, samples=samples
        )
        score_diffs = np.array(original_scores) - np.array(predicted_scores)
        score_dict[keyword.get("keyword")] = score_diffs

    score_df = pd.DataFrame(score_dict)
    # remove words with exactly the same score
    score_df = score_df.loc[:, ~score_df.T.duplicated().T]
    return score_df


def plot_scores(dataf: pd.DataFrame, concept: str = ""):
    ax = dataf.plot.box(vert=False, figsize=(12, int(dataf.shape[1] / 2.2)))
    ax.vlines(
        x=0,
        ymin=0.5,
        ymax=dataf.shape[1] + 0.5,
        colors="black",
        linestyles="dashed",
        alpha=0.5,
    )
    ax.set_title(f"Distribution of counterfactual scores for concept '{concept}'")
    ax.set_xlabel("Counterfactual scores - differences from zero indicate the direction of bias.")


def calculate_counterfactual_sample_score(sample: Sample, concept: str, clf):
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
