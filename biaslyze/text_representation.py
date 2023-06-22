"""This module contains the TextRepresentation class, which is used to represent a text in the biaslyze package."""

from typing import List, Optional

import spacy
from tqdm import tqdm

SPACY_TOKENIZER = spacy.load(
    "en_core_web_sm", disable=["parser", "tagger", "ner", "lemmatizer"]
)


class Token:
    """A class used to represent a token in the biaslyze package.

    Attributes:
        text (str): The text of the token.
        start (int): The start index of the token in the text.
        end (int): The end index of the token in the text.
        whitespace_after (str): The whitespace after the token.
    """

    def __init__(
        self,
        text: str,
        start: int,
        end: int,
        whitespace_after: str,
        function: Optional[List[str]] = None,
    ):
        """The constructor for the Token class."""
        self.text = text
        self.start = start
        self.end = end
        self.whitespace_after = whitespace_after
        self.function = function


class TextRepresentation:
    """A class used to represent a text in the biaslyze package.

    Attributes:
        text (str): The text.
        tokens (List[Token]): The tokens of the text.
    """

    def __init__(self, text: str, tokens: List[Token]):
        """The constructor for the TextRepresentation class."""
        self.text = text
        self.tokens = tokens

    def __str__(self) -> str:
        return f"TextRepresentation({self.text}, {self.tokens})"

    def __repr__(self) -> str:
        return f"TextRepresentation({self.text}, {self.tokens})"

    def __contains__(self, string: str) -> bool:
        """
        Returns True if the given string is contained in the text representation.

        Should be extended to support more complex queries.
        """
        return string in [token.text.lower() for token in self.tokens]

    @classmethod
    def from_spacy_doc(cls, doc: spacy.tokens.Doc):
        """Constructs a TextRepresentation object from a spacy doc."""
        tokens = []
        for token in doc:
            tokens.append(
                Token(token.text, token.idx, token.idx + len(token), token.whitespace_)
            )
        return cls(doc.text, tokens)


def process_texts_with_spacy(texts: List[str]) -> List[TextRepresentation]:
    """Processes the given texts with spacy."""
    # spacy_text_representations = SPACY_TOKENIZER.pipe(texts)
    return [
        TextRepresentation.from_spacy_doc(doc)
        for doc in tqdm(SPACY_TOKENIZER.pipe(texts), total=len(texts))
    ]
