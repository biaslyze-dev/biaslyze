#


## LimeKeywordBiasDetector
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/bias_detectors/lime_keyword_biasdetector.py/#L9)
```python 
LimeKeywordBiasDetector(
   n_top_keywords: int = 10,
   concept_detector: KeywordConceptDetector = KeywordConceptDetector(),
   bias_evaluator: LimeBiasEvaluator = LimeBiasEvaluator(),
   use_tokenizer: bool = False
)
```


---
Detect bias by finding keywords related to protected concepts that rank high in LIME.

Usage example:

```python
from biaslyze.bias_detectors import LimeKeywordBiasDetector

bias_detector = LimeKeywordBiasDetector(
    bias_evaluator=LimeBiasEvaluator(n_lime_samples=500),
    n_top_keywords=10
)

# detect bias in the model based on the given texts
# here, clf is a scikit-learn text classification pipeline trained for a binary classification task
detection_res = bias_detector.detect(
    texts=texts,
    predict_func=clf.predict_proba
)

# see a summary of the detection
detection_res.summary()
```


**Attributes**

* **n_top_keywords**  : In how many important LIME words should the method look for protected keywords.
* **concept_detector**  : an instance of KeywordConceptDetector
* **bias_evaluator**  : an instance of LimeBiasEvaluator
* **use_tokenizer**  : If keywords should only be searched in tokenized text. Can be useful for short keywords like 'she'.



**Methods:**


### .detect
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/bias_detectors/lime_keyword_biasdetector.py/#L55)
```python
.detect(
   texts: List[str], predict_func: Callable[[List[str]], List[float]]
)
```

---
Detect bias using keyword concept detection and lime bias evaluation.


**Args**

* **texts**  : List of texts to evaluate.
* **predict_func**  : Function that predicts a for a given text. Currently only binary classification is supported.


**Returns**

An EvaluationResults object containing the results.
