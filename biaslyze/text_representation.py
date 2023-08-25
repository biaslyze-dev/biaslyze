"""This module contains the TextRepresentation class, which is used to represent a text in the biaslyze package."""

from typing import List, Optional

import spacy
from tqdm import tqdm
from typing_extensions import Self

SPACY_TOKENIZER = spacy.load(
    "en_core_web_sm", disable=["parser", "ner", "lemmatizer"]  # "tagger"
)


class Token:
    """A class used to represent a token in the biaslyze package.

    Attributes:
        text (str): The text of the token.
        start (int): The start index of the token in the text.
        end (int): The end index of the token in the text.
        whitespace_after (str): The whitespace after the token.
        shape (str): The shape of the token as defined by spacy (e.g. Xxxx).
        function (Optional[List[str]]): The possible functions of the token (e.g. ["name", "verb"]).
    """

    def __init__(
        self,
        text: str,
        start: int,
        end: int,
        whitespace_after: str,
        shape: str,
        function: Optional[str] = None,
    ):
        """Initialize a Token."""
        self.text = text
        self.start = start
        self.end = end
        self.whitespace_after = whitespace_after
        self.shape = shape
        self.function = function

    def __str__(self) -> str:
        """Return a string representation of the Token."""
        return f"Token({self.text}, {self.start}, {self.end}, {self.whitespace_after}, {self.shape}, {self.function})"

    def __repr__(self) -> str:
        """Return a string representation of the Token."""
        return f"Token({self.text}, {self.start}, {self.end}, {self.whitespace_after}, {self.shape}, {self.function})"


class TextRepresentation:
    """A class used to represent a text in the biaslyze package.

    Attributes:
        text (str): The text.
        tokens (List[Token]): The tokens of the text.
    """

    def __init__(self, text: str, tokens: List[Token]):
        """Initialize a TextRepresentation."""
        self.text = text
        self.tokens = tokens

    def __str__(self) -> str:
        """Return a string representation of the TextRepresentation."""
        return f"TextRepresentation({self.text}, {self.tokens})"

    def __repr__(self) -> str:
        """Return a string representation of the TextRepresentation."""
        return f"TextRepresentation({self.text}, {self.tokens})"

    def __contains__(self, string: str) -> bool:
        """Check if the given string is contained in the text representation.

        Should be extended to support more complex queries.
        """
        return string.lower() in [token.text.lower() for token in self.tokens]

    @classmethod
    def from_spacy_doc(cls, doc: spacy.tokens.Doc) -> Self:
        """Construct a TextRepresentation object from a spacy doc.

        Args:
            doc (spacy.tokens.Doc): The spacy doc to construct the TextRepresentation from.

        Returns:
            TextRepresentation: The constructed TextRepresentation.
        """
        tokens = []
        for token in doc:
            tokens.append(
                Token(
                    text=token.text,
                    start=token.idx,
                    end=token.idx + len(token),
                    whitespace_after=token.whitespace_,
                    shape=token.shape_,
                    function=token.pos_,
                )
            )
        return cls(doc.text, tokens)


def process_texts_with_spacy(texts: List[str]) -> List[TextRepresentation]:
    """Process the given texts with spacy.

    Args:
        texts (List[str]): The texts to process.

    Returns:
        List[TextRepresentation]: The processed texts as TextRepresentation objects.
    """
    # spacy_text_representations = SPACY_TOKENIZER.pipe(texts)
    return [
        TextRepresentation.from_spacy_doc(doc)
        for doc in tqdm(SPACY_TOKENIZER.pipe(texts), total=len(texts))
    ]
