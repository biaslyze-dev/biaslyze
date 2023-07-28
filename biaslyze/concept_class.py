"""
This module contains the Concept class, which is used to represent a concept in the biaslyze package.
As well as Keyword Class, which is used to represent a keyword in the biaslyze package.
"""

import random
from typing import List, Optional, Tuple

from biaslyze.concepts import CONCEPTS_EN, CONCEPTS_DE
from biaslyze.text_representation import TextRepresentation, Token


class Keyword:
    """
    A class used to represent a keyword in the biaslyze package.

    Attributes:
        text (str): The word that is the keyword.
        function (List[str]): The possible functions of the keyword.
        category (str): The category of the keyword.
    """

    def __init__(self, text: str, functions: List[str], category: str):
        """The constructor for the Keyword class."""
        self.text = text
        self.functions = functions
        self.category = category

    def __str__(self) -> str:
        return f"Keyword({self.text}, {self.function}, {self.category})"

    def __repr__(self) -> str:
        return f"Keyword({self.text}, {self.function}, {self.category})"

    def can_replace_token(self, token: Token, respect_function: bool = False) -> bool:
        """Returns True if the keyword can replace the given token.

        Args:
            token (Token): The token to replace.
            respect_function (bool): Whether to respect the function of the keyword. Defaults to False.
        """
        if respect_function:
            return True
            # return token.function in self.functions
        return True

    def equal_to_token(self, token: Token) -> bool:
        """Returns True if the given token is equal to the keyword."""
        if self.text.lower() == token.text.lower():
            return True
        return False

    def get_keyword_in_style_of_token(self, token: Token) -> str:
        """Returns the keyword text in the style of the given token.

        Uses the shape of the token to determine the style.

        Args:
            token (Token): The token to get the style from.

        Returns:
            str: The keyword text in the style of the given token.
        """
        if "X" not in token.shape:
            return self.text.lower()
        elif "x" not in token.shape:
            return self.text.upper()
        elif token.shape[0] == "X":
            return self.text.capitalize()
        else:
            return self.text


class Concept:
    """
    A class used to represent a concept in the biaslyze package.

    Attributes:
        name (str): The name of the concept.
        keywords (List[Keyword]): The keywords of the concept.
    """

    def __init__(self, name: str, lang: str, keywords: List[Keyword]):
        """The constructor for the Concept class."""
        self.name = name
        self.lang = lang
        self.keywords = keywords

    @classmethod
    def from_dict_keyword_list(cls, name: str, lang: str, keywords: List[dict]):
        """Constructs a Concept object from a list of dictionaries.

        Example usage:
        ```python
        names_concept = Concept.from_dict_keyword_list(
            name="names",
            lang="de",
            keywords=[{"keyword": "Hans", "function": ["name"]}],
        )
        ```
        
        Args:
            name (str): The name of the concept.
            lang (str): The language of the concept.
            keywords (List[dict]): A list of dictionaries containing the keywords of the concept.

        """
        keyword_list = []
        for keyword in keywords:
            keyword_list.append(
                Keyword(
                    text=keyword["keyword"],
                    functions=keyword["function"],
                    category=keyword.get("category", None),
                )
            )
        return cls(name, lang, keyword_list)

    def get_present_keywords(
        self, text_representation: TextRepresentation
    ) -> List[Keyword]:
        """Returns the keywords that are present in the given text."""
        present_keywords = []
        for keyword in self.keywords:
            if keyword.text in text_representation:
                present_keywords.append(keyword)
        return present_keywords

    def get_counterfactual_texts(
        self,
        keyword: Keyword,
        text_representation: TextRepresentation,
        n_texts: Optional[int] = None,
        respect_function: bool = True,
    ) -> List[Tuple[str, Keyword]]:
        """Returns a counterfactual texts based on a specific keyword for the given text representation.

        Args:
            keyword (Keyword): The keyword in the text to replace.
            text_representation (TextRepresentation): The text representation to replace the keyword in.
            n_texts (Optional[int]): The number of counterfactual texts to return. Defaults to None, which returns all possible counterfactual texts.
            respect_function (bool): Whether to respect the function of the keyword. Defaults to True.

        Returns:
            List[Tuple[str, Keyword]]: A list of tuples containing the counterfactual text and the keyword that was replaced.
        """
        counterfactual_texts = []
        for token in text_representation.tokens:
            # check if the token is equal to the keyword
            if keyword.equal_to_token(token):
                # shuffle the keywords
                random.shuffle(self.keywords)
                # create a counterfactual text for each keyword until n_texts is reached
                for counterfactual_keyword in self.keywords:
                    # check if the keyword can be replaced by another keyword
                    if counterfactual_keyword.can_replace_token(
                        token, respect_function
                    ):
                        # create the counterfactual text
                        counterfactual_text = (
                            text_representation.text[: token.start]
                            + counterfactual_keyword.get_keyword_in_style_of_token(
                                token
                            )
                            + text_representation.text[token.end :]
                        )
                        counterfactual_texts.append(
                            (counterfactual_text, counterfactual_keyword)
                        )
                    # check if n_texts is reached and return the counterfactual texts
                    if n_texts and (len(counterfactual_texts) >= n_texts):
                        return counterfactual_texts
        return counterfactual_texts


def load_concepts(lang: str) -> List[Concept]:
    """Loads the concepts from the concepts.py file.

    Args:
        lang (str): The language of the concepts to load.

    TODO:
    - Make this load from a JSON file instead of a Python file.
    """
    concept_list = []
    if lang == "en":
        for concept_name, concept_keywords in CONCEPTS_EN.items():
            concept_list.append(
                Concept.from_dict_keyword_list(concept_name, lang, concept_keywords)
            )
    elif lang == "de":
        for concept_name, concept_keywords in CONCEPTS_DE.items():
            concept_list.append(
                Concept.from_dict_keyword_list(concept_name, lang, concept_keywords)
            )
    else:
        raise ValueError(f"Language {lang} not supported.")
    return concept_list
