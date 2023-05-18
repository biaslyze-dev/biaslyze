#


## CounterfactualBiasDetector
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/bias_detectors/counterfactual_biasdetector.py/#L19)
```python 
CounterfactualBiasDetector(
   use_tokenizer: bool = False,
   concept_detector: KeywordConceptDetector = KeywordConceptDetector()
)
```


---
Detect hints of bias by calculating counterfactual token scores for protected concepts.

Usage example:

```python
from biaslyze.bias_detectors import CounterfactualBiasDetector

bias_detector = CounterfactualBiasDetector()

# detect bias in the model based on the given texts
# here, clf is a scikit-learn text classification pipeline trained for a binary classification task
detection_res = bias_detector.process(
    texts=texts,
    predict_func=clf.predict_proba
)

# see a summary of the detection
detection_res.report()

# visualize the counterfactual scores
detection_res.visualize_counterfactual_scores()
```


**Attributes**

* **use_tokenizer**  : If keywords should only be searched in tokenized text. Can be useful for short keywords like 'she'.
* **concept_detector**  : an instance of KeywordConceptDetector



**Methods:**


### .process
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/bias_detectors/counterfactual_biasdetector.py/#L59)
```python
.process(
   texts: List[str], predict_func: Callable[[List[str]], List[float]],
   concepts_to_consider: List = None, max_counterfactual_samples: int = None
)
```

---
Detect bias by masking out words.


**Args**

* **texts**  : texts to probe the model for bias.
* **predict_func**  : Function to run the texts through the model and get probabilities as outputs.
* **concepts_to_consider**  : If given, only the given concepts are considered.
* **max_counterfactual_samples**  : If given, only the given number of counterfactual samples are used for each concept.


**Returns**

A [CounterfactualDetectionResult](/biaslyze/results/counterfactual_detection_results/) object.
