#


## Token
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/text_representation.py/#L13)
```python 
Token(
   text: str, start: int, end: int, whitespace_after: str,
   function: Optional[List[str]] = None
)
```


---
A class used to represent a token in the biaslyze package.


**Attributes**

* **text** (str) : The text of the token.
* **start** (int) : The start index of the token in the text.
* **end** (int) : The end index of the token in the text.
* **whitespace_after** (str) : The whitespace after the token.


----


## TextRepresentation
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/text_representation.py/#L39)
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
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/text_representation.py/#L67)
```python
.from_spacy_doc(
   cls, doc: spacy.tokens.Doc
)
```

---
Constructs a TextRepresentation object from a spacy doc.
