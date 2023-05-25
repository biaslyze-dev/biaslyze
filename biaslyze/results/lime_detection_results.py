"""Classes to return results of the different steps."""
from collections import Counter, defaultdict
from pprint import pprint
from typing import Dict, List

from biaslyze._plotting import _plot_histogram_dashboard


class LimeSampleResult:
    """A sample on which the model might behave biased based on LIME.

    Contains details on why it might be biased and what concepts are affected.

    Attributes:
        text: Sample text
        bias_concepts: Protected concepts present in the text by which the model appears biased.
        bias_reasons: Reasons why bias was detected. Might be a list of keywords.
        top_words: Most important words for the prediction.
        num_tokens: Number of unique tokens in the text.
        keyword_position: Position of the keyword in the top_words list.
        score: Score of the sample.
        metrics: Metrics of the LIME explainer.
    """

    def __init__(
        self,
        text: str,
        bias_concepts: List[str],
        bias_reasons: List[str],
        top_words: List[str],
        num_tokens: List[str],
        keyword_position: int,
        score: float,
        metrics: Dict = None,
    ):
        self.text = text
        self.bias_concepts = bias_concepts
        self.bias_reasons = bias_reasons
        self.top_words = top_words
        self.num_tokens = num_tokens
        self.keyword_position = keyword_position
        self.score = score
        self.metrics = metrics

    def __repr__(self) -> str:
        return f"''{self.text}'' might contain bias {self.bias_concepts}; reasons: {self.bias_reasons}"


class LimeDetectionResult:
    """Contains all samples on detected potential bias issues.

    Attributes:
        biased_samples: A list of BiasedSampleResults.
    """

    def __init__(self, biased_samples: List[LimeSampleResult]):
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
                    concept_groups[bias_concept].append(
                        {"text": sample.text, "reason": sample.bias_reasons}
                    )
            for concept, group in concept_groups.items():
                print(f"Concept: {concept}")
                pprint(group)
        else:
            for sample in self.biased_samples:
                print(sample)

    def __repr__(self) -> str:
        """Return a string representation of the EvaluationResult."""
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

    def dashboard(self, use_position=False):
        """Return a bokeh dashboard.

        Content of the dashboard:
            - A histogram of the LIME score of the samples.
            - A table with the details of the samples.

        Args:
            use_position: If True, use the normalized position for plotting.
        """

        dashboard = _plot_histogram_dashboard(
            texts=[sample.text for sample in self.biased_samples],
            concepts=[sample.bias_concepts for sample in self.biased_samples],
            scores=[sample.score for sample in self.biased_samples],
            keywords=[sample.bias_reasons for sample in self.biased_samples],
            keyword_positions=[
                sample.keyword_position for sample in self.biased_samples
            ],
            num_tokens=[sample.num_tokens for sample in self.biased_samples],
            top_words=[sample.top_words for sample in self.biased_samples],
            use_position=use_position,
            score_version="LimeScore",
        )

        return dashboard
