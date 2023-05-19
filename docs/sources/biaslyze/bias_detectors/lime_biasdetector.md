#


## LimeBiasDetector
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/bias_detectors/lime_biasdetector.py/#L18)
```python 
LimeBiasDetector(
   n_lime_samples: int = 1000, use_tokenizer: bool = False,
   concept_detector: KeywordConceptDetector = KeywordConceptDetector()
)
```


---
Detect bias by finding keywords related to protected concepts that rank high in LIME.

Usage example:

```python
from biaslyze.bias_detectors import LimeKeywordBiasDetector

bias_detector = LimeKeywordBiasDetector(
    n_lime_samples=500,
)

# detect bias in the model based on the given texts
# here, clf is a scikit-learn text classification pipeline trained for a binary classification task
detection_res = bias_detector.detect(
    texts=texts,
    predict_func=clf.predict_proba,
    n_top_keywords=10,
)

# see a summary of the detection
detection_res.summary()
```


**Attributes**

* **n_lime_samples**  : Number of perturbed samples to create for each LIME run.
* **use_tokenizer**  : If keywords should only be searched in tokenized text. Can be useful for short keywords like 'she'.
* **concept_detector**  : An instance of KeywordConceptDetector



**Methods:**


### .detect
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/bias_detectors/lime_biasdetector.py/#L66)
```python
.detect(
   texts: List[str], predict_func: Callable[[List[str]], List[float]],
   top_n_keywords: int = 10
)
```

---
Detect bias using keyword concept detection and lime bias evaluation.


**Args**

* **texts**  : List of texts to evaluate.
* **predict_func**  : Function that predicts a for a given text. Currently only binary classification is supported.
* **top_n_keywords**  : How many keywords detected by LIME should be considered for bias detection.


**Returns**

A LimeDetectionResult containing all samples with detected bias.
