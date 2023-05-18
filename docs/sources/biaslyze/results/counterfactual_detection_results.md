#


## CounterfactualDetectionResult
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/counterfactual_detection_results.py/#L55)
```python 
CounterfactualDetectionResult(
   concept_results: List[CounterfactualConceptResult]
)
```


---
The result of a counterfactual bias detection run.


**Attributes**

* **concept_results**  : A list of CounterfactualConceptResult objects.



**Methods:**


### .report
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/counterfactual_detection_results.py/#L78)
```python
.report()
```

---
Show an overview of the results.

Details:
For each concept, the maximum mean and maximum standard deviation of the counterfactual scores is shown.

### .visualize_counterfactual_scores
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/counterfactual_detection_results.py/#L89)
```python
.visualize_counterfactual_scores(
   concept: str, top_n: int = None
)
```

---
Visualize the counterfactual scores for a given concept.

The score is calculated by comparing the prediction of the original sample with the prediction of the counterfactual sample.
For every keyword shown all concept keywords are replaced with the keyword shown. The difference in the prediction to the original is then calculated.


**Args**

* **concept**  : The concept to visualize.
* **top_n**  : If given, only the top n keywords are shown.

---
Details:
    The counterfactual scores are shown as boxplots. The median of the scores is indicated by a dashed line.

### .visualize_counterfactual_sample_scores
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/counterfactual_detection_results.py/#L127)
```python
.visualize_counterfactual_sample_scores(
   concept: str, top_n: int = None
)
```

---
Visualize the counterfactual scores given concept.

This differs from visualize_counterfactual_score_by_sample in that it shows the counterfactual
score grouped by the original keyword in the text, not the counterfactual keyword.


**Args**

* **concept**  : The concept to visualize.
* **top_n**  : If given, only the top n keywords are shown.


### .visualize_counterfactual_score_by_sample
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/counterfactual_detection_results.py/#L179)
```python
.visualize_counterfactual_score_by_sample(
   concept: str
)
```

---
Visualize the counterfactual scores for each sample for a given concept.

----


## CounterfactualConceptResult
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/counterfactual_detection_results.py/#L39)
```python 
CounterfactualConceptResult(
   concept: str, scores: pd.DataFrame, omitted_keywords: List[str],
   counterfactual_samples: List[CounterfactualSample] = None
)
```


---
The result of a counterfactual bias detection run for a single concept.

----


## CounterfactualSample
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/counterfactual_detection_results.py/#L10)
```python 
CounterfactualSample(
   text: str, orig_keyword: str, keyword: str, concept: str, tokenized: List[str]
)
```


---
A sample for counterfactual bias detection.


**Attributes**

* **text**  : The original text.
* **orig_keyword**  : The original keyword.
* **keyword**  : The keyword that replaced the original keyword.
* **concept**  : The concept that was detected in the text.
* **tokenized**  : The tokenized text in spacy representation.

