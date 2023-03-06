#


## LimeKeywordBiasDetector
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/bias_detectors/lime_keyword_biasdetector.py/#L11)
```python 
LimeKeywordBiasDetector(
   predict_func: Callable[[List[str]], List[float]], n_top_keywords: int = 10,
   concept_detector = KeywordConceptDetector(),
   bias_evaluator = LimeBiasEvaluator()
)
```


---
Detect bias by finding keywords related to protected concepts that rank high in LIME.

Usage example:

```python
from biaslyze.bias_detectors import LimeKeywordBiasDetector

bias_detector = LimeKeywordBiasDetector(
    predict_func=clf.predict_proba,    # here, clf is a scikit-learn text classification pipeline trained for a binary classification task
    bias_evaluator=LimeBiasEvaluator(n_lime_samples=500),
    n_top_keywords=10
)

# detect bias in the model based on the given texts
detection_res = bias_detector.detect(texts)

# see a summary of the detection
detection_res.summary()
```


**Attributes**

* **predict_func**  : Function that predicts a for a given text. Currently only binary classification is supported.
* **n_top_keywords**  : In how many important LIME words should the method look for protected keywords.
* **concept_detector**  : an instance of KeywordConceptDetector 
* **bias_evaluator**  : an instance of LimeBiasEvaluator



**Methods:**


### .detect
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/bias_detectors/lime_keyword_biasdetector.py/#L51)
```python
.detect(
   texts: List[str]
)
```

---
Detect bias using keyword concept detection and lime bias evaluation.


**Args**

* **texts**  : List of texts to evaluate.


**Returns**

An EvaluationResults object containing the results.
