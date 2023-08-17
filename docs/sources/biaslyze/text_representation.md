#


## Token
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/text_representation.py/#L14)
```python 
Token(
   text: str, start: int, end: int, whitespace_after: str, shape: str,
   function: Optional[str] = None
)
```


---
A class used to represent a token in the biaslyze package.


**Attributes**

* **text** (str) : The text of the token.
* **start** (int) : The start index of the token in the text.
* **end** (int) : The end index of the token in the text.
* **whitespace_after** (str) : The whitespace after the token.
* **shape** (str) : The shape of the token as defined by spacy (e.g. Xxxx).
* **function** (Optional[List[str]]) : The possible functions of the token (e.g. ["name", "verb"]).


----


## TextRepresentation
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/text_representation.py/#L44)
```python 
TextRepresentation(
   text: str, tokens: List[Token]
)
```


---
A class used to represent a text in the biaslyze package.


**Attributes**

* **text** (str) : The text.
* **tokens** (List[Token]) : The tokens of the text.



**Methods:**


### .from_spacy_doc
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/text_representation.py/#L73)
```python
.from_spacy_doc(
   cls, doc: spacy.tokens.Doc
)
```

---
Construct a TextRepresentation object from a spacy doc.


**Args**

* **doc** (spacy.tokens.Doc) : The spacy doc to construct the TextRepresentation from.


**Returns**

* **TextRepresentation**  : The constructed TextRepresentation.

