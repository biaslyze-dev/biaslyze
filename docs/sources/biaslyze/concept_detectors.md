#


## KeywordConceptDetector
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/concept_detectors.py/#L11)
```python 
KeywordConceptDetector(
   lang: str = 'en', use_tokenizer: bool = False
)
```


---
Use keywords to determine if a protected concept is present in text.


**Attributes**

* **lang**  : The language of the text. Currently only 'en' and 'de' are supported.
* **use_tokenizer**  : If keywords should only be searched in tokenized text. Can be useful for short keywords like 'she'.


**Raises**

* **ValueError**  : If the language is not supported.



**Methods:**


### .detect
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/concept_detectors.py/#L35)
```python
.detect(
   texts: List[str], concepts_to_consider: Optional[List[str]] = None
)
```

---
Detect concepts present in texts.

Returns a list of texts with the concept present.


**Args**

* **texts**  : List of texts to look for protected concepts.
* **concepts_to_consider**  : List of concepts to consider. If None, all concepts are considered.


**Returns**

List of texts where protected concepts are detected.
