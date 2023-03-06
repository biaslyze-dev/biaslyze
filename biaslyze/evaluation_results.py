"""Classes to return results of the different steps."""
from typing import List
from collections import Counter


class BiasedSampleResult:
    def __init__(self, text: str, bias_concepts: List[str], bias_reasons: List[str]):
        self.text = text
        self.bias_concepts = bias_concepts
        self.bias_reasons = bias_reasons

    def __repr__(self) -> str:
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
    Potentially problematic concepts detected: {concepts_stats}
    Based on keywords: {reasons_stats}."""
        return representation_string
