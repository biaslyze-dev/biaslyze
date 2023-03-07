"""Classes to return results of the different steps."""
from collections import Counter, defaultdict
from typing import List
from pprint import pprint


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

    def details(self, group_by_concept: bool = False):
        """Print the details of every biased sample detected.
        
        Args:
            group_by_concept: If the output should be grouped by concepts.
        """
        if group_by_concept:
            concept_groups = defaultdict(list)
            for sample in self.biased_samples:
                for bias_concept in sample.bias_concepts:
                    concept_groups[bias_concept].append({"text": sample.text, "reason": sample.bias_reasons})
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
