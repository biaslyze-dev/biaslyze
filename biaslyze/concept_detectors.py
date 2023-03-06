"""This module contains classes to detect the presence of protected concepts in texts."""
from typing import List, Union

from biaslyze.concepts import CONCEPTS


class KeywordConceptDetector:
    def __init__(self):
        pass

    def detect(self, texts: List[str], labels: List) -> Union[List[str], List]:
        """Detect concepts present in texts.

        Returns a list of texts with the concept present.
        """
        detected_texts = []
        detected_labels = []
        for text, label in zip(texts, labels):
            if any(n in text.lower() for n in CONCEPTS.get("nationality")):
                detected_texts.append(text)
                detected_labels.append(label)
        return detected_texts, detected_labels
