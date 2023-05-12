"""This module contains classes to detect the presence of protected concepts in texts."""
from typing import List

import spacy
from loguru import logger
from tqdm import tqdm

from biaslyze.concepts import CONCEPTS


class KeywordConceptDetector:
    """Use keywords to determine if a protected concept is present in text.

    Attributes:
        use_tokenizer: If keywords should only be searched in tokenized text. Can be useful for short keywords like 'she'.
    """

    def __init__(self, use_tokenizer: bool = False):
        self.use_tokenizer = use_tokenizer
        self._tokenizer = spacy.load(
            "en_core_web_sm", disable=["parser", "tagger", "ner", "lemmatizer"]
        )

    def detect(self, texts: List[str]) -> List[str]:
        """Detect concepts present in texts.

        Returns a list of texts with the concept present.

        Args:
            texts: List of texts to look for protected concepts.

        Returns:
            List of texts where protected concepts are detected.
        """
        logger.info(f"Started keyword-based concept detection on {len(texts)} texts...")
        detected_texts = []
        concept_keywords = [
            keyword["keyword"]
            for concept_keywords in CONCEPTS.values()
            for keyword in concept_keywords
        ]
        for text in tqdm(texts):
            if self.use_tokenizer:
                text_representation = [
                    token.text.lower() for token in self._tokenizer(text)
                ]
            else:
                text_representation = text.lower()
            if any(keyword in text_representation for keyword in concept_keywords):
                detected_texts.append(text)
        logger.info(f"Done. Found {len(detected_texts)} texts with protected concepts.")
        return detected_texts
