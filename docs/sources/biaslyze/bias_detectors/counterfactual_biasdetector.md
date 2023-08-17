#


## CounterfactualBiasDetector
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/bias_detectors/counterfactual_biasdetector.py/#L19)
```python 
CounterfactualBiasDetector(
   lang: str = 'en', use_tokenizer: bool = False
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

# visualize the counterfactual scores as a dash dashboard
detection_res.dashboard()
```


**Attributes**

* **lang**  : The language of the texts. Decides which concepts and keywords to use.
* **use_tokenizer**  : If keywords should only be searched in tokenized text. Can be useful for short keywords like 'she'.



**Methods:**


### .register_concept
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/bias_detectors/counterfactual_biasdetector.py/#L70)
```python
.register_concept(
   concept: Concept
)
```

---
Register a new, custom concept to the detector.

Example usage:
```python
names_concept = Concept.from_dict_keyword_list(
name="names",
lang="de",
keywords=[{"keyword": "Hans", "function": ["name"]}],
---
)
bias_detector = CounterfactualBiasDetector(lang="de")
bias_detector.register_concept(names_concept)
```


**Args**

* **concept**  : The concept to register.


**Raises**

* **ValueError**  : If concept is not a Concept object.
* **ValueError**  : If a concept with this name is already registered.


### .process
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/bias_detectors/counterfactual_biasdetector.py/#L97)
```python
.process(
   texts: List[str], predict_func: Callable[[List[str]], List[float]],
   labels: Optional[List[str]] = None, concepts_to_consider: Optional[List[str]] = [],
   max_counterfactual_samples: Optional[int] = None,
   max_counterfactual_samples_per_text: Optional[int] = None
)
```

---
Detect potential bias in the model based on the given texts.


**Args**

* **texts**  : texts to probe the model for bias.
* **predict_func**  : Function to run the texts through the model and get probabilities as outputs.
* **labels**  : Optional. Used to add labels to the counterfactual results.
* **concepts_to_consider**  : If given, only the given concepts are considered.
* **max_counterfactual_samples**  : Optional. The maximum number of counterfactual samples to return. Defaults to None, which returns all possible counterfactual samples.
* **max_counterfactual_samples_per_text**  : Optional. The maximum number of counterfactual samples to return per text. Defaults to None, which returns all possible counterfactual samples.


**Returns**

A [CounterfactualDetectionResult](/biaslyze/results/counterfactual_detection_results/) object.


**Raises**

* **ValueError**  : If texts or predict_func is not given.
* **ValueError**  : If concepts_to_consider is not a list.
* **ValueError**  : If max_counterfactual_samples is given but not a positive integer.
* **ValueError**  : If max_counterfactual_samples_per_text is given but not a positive integer.
* **ValueError**  : If concepts_to_consider contains a concept that is not registered.

