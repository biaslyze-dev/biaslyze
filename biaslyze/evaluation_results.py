"""Classes to return results of the different steps."""
from collections import Counter
from typing import List


class BiasedSampleResult:
    """A sample on which the model might behave biased.
    
    Contains details on why it might be biased and what concepts are affected.
    
    Attributes:
        text: Sample text
        bias_concepts: Protected concepts present in the text by which the model appears biased.
        bias_reasons: Reasons why bias was detected. Might be a list of keywords.
    """
    def __init__(self, text: str, bias_concepts: List[str], bias_reasons: List[str]):
        self.text = text
        self.bias_concepts = bias_concepts
        self.bias_reasons = bias_reasons

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
