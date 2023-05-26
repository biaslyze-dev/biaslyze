#


## CounterfactualBiasDetector
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/bias_detectors/counterfactual_biasdetector.py/#L20)
```python 
CounterfactualBiasDetector(
   use_tokenizer: bool = False,
   concept_detector: KeywordConceptDetector = KeywordConceptDetector()
)
```


---
Detect hints of bias by calculating counterfactual token scores for protected concepts.

The counterfactual score is defined as the difference between the predicted
probability score for the original text and the predicted probability score for the counterfactual text.

$$counterfactual_score = P(x=1|counterfactual_text) - P(x=1|original_text),$$

where counterfactual text is defined as the original text where a keyword of the given concept is
replaced by another keyword of the same concept. So a counterfactual_score > 0 means that the
model is more likely to predict the positive class for the original text than for the counterfactual text.

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
detection_res.visualize_counterfactual_scores(concept="religion")

# visualize the counterfactual sample scores
detection_res.visualize_counterfactual_score_by_sample_histogram(concepts=["religion", "gender"])
```


**Attributes**

* **use_tokenizer**  : If keywords should only be searched in tokenized text. Can be useful for short keywords like 'she'.
* **concept_detector**  : an instance of KeywordConceptDetector



**Methods:**


### .process
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/bias_detectors/counterfactual_biasdetector.py/#L72)
```python
.process(
   texts: List[str], predict_func: Callable[[List[str]], List[float]],
   labels: Optional[List[str]] = None, concepts_to_consider: Optional[List[str]] = [],
   max_counterfactual_samples: Optional[int] = None
)
```

---
Detect potential bias in the model based on the given texts.


**Args**

* **texts**  : texts to probe the model for bias.
* **predict_func**  : Function to run the texts through the model and get probabilities as outputs.
* **labels**  : Optional. Used to add labels to the counterfactual results.
* **concepts_to_consider**  : If given, only the given concepts are considered.
* **max_counterfactual_samples**  : If given, only the given number of counterfactual samples are used for each concept.


**Returns**

A [CounterfactualDetectionResult](/biaslyze/results/counterfactual_detection_results/) object.


**Raises**

* **ValueError**  : If texts or predict_func is not given.

