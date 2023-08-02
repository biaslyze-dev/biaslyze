#


## Concept
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/concept_class.py/#L71)
```python 
Concept(
   name: str, lang: str, keywords: List[Keyword]
)
```


---
A class used to represent a concept in the biaslyze package.

Currently the following concepts are supported:

in English:

- gender
- religion
- ethnicity
- gendered_words
- nationality

in German:

- gender
- religion

You can find out more here: [concepts.py](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/concepts.py).


**Attributes**

* **name** (str) : The name of the concept.
* **keywords** (List[Keyword]) : The keywords of the concept.



**Methods:**


### .from_dict_keyword_list
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/concept_class.py/#L104)
```python
.from_dict_keyword_list(
   cls, name: str, lang: str, keywords: List[dict]
)
```

---
Constructs a Concept object from a list of dictionaries.

Example usage:
```python
names_concept = Concept.from_dict_keyword_list(
name="names",
lang="de",
keywords=[{"keyword": "Hans", "function": ["name"]}],
---
)
```


**Args**

* **name** (str) : The name of the concept.
* **lang** (str) : The language of the concept.
* **keywords** (List[dict]) : A list of dictionaries containing the keywords of the concept.


### .get_present_keywords
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/concept_class.py/#L133)
```python
.get_present_keywords(
   text_representation: TextRepresentation
)
```

---
Returns the keywords that are present in the given text.

### .get_counterfactual_texts
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/concept_class.py/#L143)
```python
.get_counterfactual_texts(
   keyword: Keyword, text_representation: TextRepresentation,
   n_texts: Optional[int] = None, respect_function: bool = True
)
```

---
Returns a counterfactual texts based on a specific keyword for the given text representation.


**Args**

* **keyword** (Keyword) : The keyword in the text to replace.
* **text_representation** (TextRepresentation) : The text representation to replace the keyword in.
* **n_texts** (Optional[int]) : The number of counterfactual texts to return. Defaults to None, which returns all possible counterfactual texts.
* **respect_function** (bool) : Whether to respect the function of the keyword. Defaults to True.


**Returns**

* A list of tuples containing the counterfactual text and the keyword that was replaced.


----


## Keyword
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/concept_class.py/#L10)
```python 
Keyword(
   text: str, functions: List[str], category: str
)
```


---
A class used to represent a keyword in the biaslyze package.


**Attributes**

* **text** (str) : The word that is the keyword.
* **function** (List[str]) : The possible functions of the keyword.
* **category** (str) : The category of the keyword.



**Methods:**


### .can_replace_token
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/concept_class.py/#L32)
```python
.can_replace_token(
   token: Token, respect_function: bool = False
)
```

---
Returns True if the keyword can replace the given token.


**Args**

* **token** (Token) : The token to replace.
* **respect_function** (bool) : Whether to respect the function of the keyword. Defaults to False.


### .equal_to_token
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/concept_class.py/#L44)
```python
.equal_to_token(
   token: Token
)
```

---
Returns True if the given token is equal to the keyword.

### .get_keyword_in_style_of_token
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/concept_class.py/#L50)
```python
.get_keyword_in_style_of_token(
   token: Token
)
```

---
Returns the keyword text in the style of the given token.

Uses the shape of the token to determine the style.


**Args**

* **token** (Token) : The token to get the style from.


**Returns**

* **str**  : The keyword text in the style of the given token.

