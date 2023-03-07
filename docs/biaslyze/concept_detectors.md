#


## KeywordConceptDetector
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/concept_detectors.py/#L10)
```python 
KeywordConceptDetector(
   use_tokenizer: bool = False
)
```


---
Use keywords to determine if a protected concept is present in text.


**Attributes**

* **use_tokenizer**  : If keywords should only be searched in tokenized text. Can be useful for short keywords like 'she'.



**Methods:**


### .detect
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/concept_detectors.py/#L21)
```python
.detect(
   texts: List[str]
)
```

---
Detect concepts present in texts.

Returns a list of texts with the concept present.


**Args**

* **texts**  : List of texts to look for protected concepts.


**Returns**

List of texts where protected concepts are detected.
