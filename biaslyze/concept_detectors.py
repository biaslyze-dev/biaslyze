"""This module contains classes to detect the presence of protected concepts in texts."""
from typing import List, Optional

import spacy
from loguru import logger
from tqdm import tqdm

from biaslyze.concepts.concepts_de import CONCEPTS_DE
from biaslyze.concepts.concepts_en import CONCEPTS_EN


class KeywordConceptDetector:
    """Use keywords to determine if a protected concept is present in text.

    Attributes:
        lang: The language of the text. Currently only 'en' and 'de' are supported.
        use_tokenizer: If keywords should only be searched in tokenized text. Can be useful for short keywords like 'she'.

    Raises:
        ValueError: If the language is not supported.
    """

    def __init__(self, lang: str = "en", use_tokenizer: bool = False):
        """Initialize the KeywordConceptDetector."""
        lang = lang
        self.use_tokenizer = use_tokenizer
        self._tokenizer = spacy.load(
            "en_core_web_sm", disable=["parser", "tagger", "ner", "lemmatizer"]
        )
        if lang == "en":
            self.concepts = CONCEPTS_EN
        elif lang == "de":
            self.concepts = CONCEPTS_DE
        else:
            raise ValueError(f"Language {lang} not supported.")

    def detect(
        self, texts: List[str], concepts_to_consider: Optional[List[str]] = None
    ) -> List[str]:
        """Detect concepts present in texts.

        Returns a list of texts with the concept present.

        Args:
            texts: List of texts to look for protected concepts.
            concepts_to_consider: List of concepts to consider. If None, all concepts are considered.

        Returns:
            List of texts where protected concepts are detected.
        """
        logger.info(f"Started keyword-based concept detection on {len(texts)} texts...")
        detected_texts = []
        concept_keywords = [
            keyword["keyword"]
            for concept_name, concept_keywords in self.concepts.items()
            for keyword in concept_keywords
            if (concepts_to_consider is None) or (concept_name in concepts_to_consider)
        ]
        for text in tqdm(texts):
            if self.use_tokenizer:
                text_representation: List[str] = [
                    token.text.lower() for token in self._tokenizer(text)
                ]
            else:
                text_representation: str = text.lower()
            if any(keyword in text_representation for keyword in concept_keywords):
                detected_texts.append(text)
        logger.info(f"Done. Found {len(detected_texts)} texts with protected concepts.")
        return detected_texts
