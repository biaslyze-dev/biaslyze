#


## CounterfactualDetectionResult
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/counterfactual_detection_results.py/#L65)
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
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/counterfactual_detection_results.py/#L111)
```python
.report()
```

---
Show an overview of the results.

Details:
For each concept, the maximum mean and maximum standard deviation of the counterfactual scores is shown.

### .dashboard
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/counterfactual_detection_results.py/#L122)
```python
.dashboard(
   num_keywords: int = 10
)
```

---
Start a dash dashboard with interactive box plots.

### .visualize_counterfactual_score_by_sample
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/counterfactual_detection_results.py/#L252)
```python
.visualize_counterfactual_score_by_sample(
   concept: str
)
```

---
Visualize the counterfactual scores for each sample for a given concept.

----


## CounterfactualConceptResult
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/counterfactual_detection_results.py/#L49)
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
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/counterfactual_detection_results.py/#L14)
```python 
CounterfactualSample(
   text: str, orig_keyword: str, keyword: str, concept: str, tokenized: List[str],
   label: int = None, source_text: str = None
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
* **label**  : The label of the original text.
* **source_text**  : The source text from which the text was derived.

