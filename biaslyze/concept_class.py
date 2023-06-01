"""
This module contains the Concept class, which is used to represent a concept in the biaslyze package.
As well as Keyword Class, which is used to represent a keyword in the biaslyze package.
"""

from typing import List


class Keyword:
    """
    A class used to represent a keyword in the biaslyze package.

    Attributes:
        word (str): The word that is the keyword.
        function (List[str]): The possible functions of the keyword.
        category (str): The category of the keyword.
    """

    def __init__(self, word: str, function: List[str], category: str):
        """The constructor for the Keyword class."""
        self.word = word
        self.function = function
        self.category = category

    def __str__(self) -> str:
        return f"Keyword({self.word}, {self.function}, {self.category})"

    def __repr__(self) -> str:
        return f"Keyword({self.word}, {self.function}, {self.category})"


class Concept:
    """
    A class used to represent a concept in the biaslyze package.

    Attributes:
        name (str): The name of the concept.
        keywords (List[Keyword]): The keywords of the concept.        
    """

    def __init__(self, name: str, keywords: List[Keyword]):
        """The constructor for the Concept class."""
        self.name = name
        self.keywords = keywords

    @classmethod
    def from_dict_keyword_list(cls, name: str, keywords: List[dict]):
        """Constructs a Concept object from a list of dictionaries."""
        keyword_list = []
        for keyword in keywords:
            keyword_list.append(Keyword(keyword["keyword"], keyword["function"], keyword["category"]))
        return cls(name, keyword_list)