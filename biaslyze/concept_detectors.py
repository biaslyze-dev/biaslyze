"""This module contains classes to detect the presence of protected concepts in texts."""
from typing import List

from biaslyze.concepts import CONCEPTS


class KeywordConceptDetector:
    """Use keywords to determine if a protected concept is present in text."""

    def detect(self, texts: List[str]) -> List[str]:
        """Detect concepts present in texts.

        Returns a list of texts with the concept present.

        Args:
            texts: List of texts to look for protected concepts.

        Returns:
            List of texts where protected concepts are detected.
        """
        detected_texts = []
        concept_keywords = [
            keyword
            for concept_keywords in CONCEPTS.values()
            for keyword in concept_keywords
        ]
        for text in texts:
            if any(keyword in text.lower() for keyword in concept_keywords):
                detected_texts.append(text)
        return detected_texts
