"""Contains classes to evaluate the bias of detected concepts."""
from typing import List
import numpy as np
from tqdm import tqdm
import warnings
from eli5.lime import TextExplainer

from concepts import CONCEPTS


class LimeBiasEvaluator:
    def __init__(self, n_lime_samples: int = 100):
        self.n_lime_samples = n_lime_samples
        self.explainer = TextExplainer(n_samples=n_lime_samples)

    def evaluate(self, predict_func, texts: List[str], top_n: int = 10):
        """Evaluate if a bias is present."""
        warnings.filterwarnings("ignore", category=FutureWarning)
        biased_samples = []
        for text in tqdm(texts):
            self.explainer.fit(text, predict_func)
            interpret_sample_dict = {
                coef: token
                for coef, token in zip(
                    self.explainer.clf_.coef_[0],
                    self.explainer.vec_.get_feature_names_out(),
                )
            }
            top_interpret_sample_dict = sorted(
                interpret_sample_dict.items(), key=lambda x: -np.abs(x[0])
            )[: min(len(interpret_sample_dict), top_n)]
            important_tokens = [w.lower() for (_, w) in top_interpret_sample_dict]

            # check for concepts reasons
            bias_indicator_tokens = []
            bias_concepts = []
            for concept, concept_keywords in CONCEPTS.items():
                biased_tokens_set = set(concept_keywords).intersection(
                    set(important_tokens)
                )
                if len(biased_tokens_set) > 0:
                    bias_concepts.append(concept)
                    bias_indicator_tokens.extend(list(biased_tokens_set))

            if len(bias_concepts) > 0:
                biased_samples.append(
                    BiasedSampleResult(text, bias_concepts, bias_indicator_tokens)
                )

        return EvaluationResult(biased_samples)


class BiasedSampleResult:
    def __init__(self, text: str, bias_concepts: List[str], bias_reasons: List[str]):
        self.text = text
        self.bias_concepts = bias_concepts
        self.bias_reasons = bias_reasons

    def __repr__(self):
        return f"''{self.text}'' might contain bias {self.bias_concepts}; reasons: {self.bias_reasons}"


class EvaluationResult:
    def __init__(self, biased_samples: List[BiasedSampleResult]):
        self.biased_samples = biased_samples

    def summary(self):
        print(self.__repr__())

    def details(self):
        """Print the details of every biased sample detected."""
        for sample in self.biased_samples:
            print(sample)

    def __repr__(self):
        concepts = []
        reasons = []
        for sample in self.biased_samples:
            concepts.extend(sample.bias_concepts)
            reasons.extend(sample.bias_reasons)
        representation_string = f"""Detected {len(self.biased_samples)} samples with potential issues.
Potentially problematic concepts detected: {set(concepts)}
Based on keywords: {set(reasons)}."""
        return representation_string
