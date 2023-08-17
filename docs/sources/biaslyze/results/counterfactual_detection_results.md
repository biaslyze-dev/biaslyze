#


## CounterfactualDetectionResult
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/counterfactual_detection_results.py/#L81)
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


### .save
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/counterfactual_detection_results.py/#L92)
```python
.save(
   path: str
)
```

---
Save the detection result to a file.

Load again by

```python
from biaslyze.utils import load_results

results = load_results(path)
```


**Args**

* **path** (str) : The path to save the result to.


**Raises**

* **ValueError**  : If the path is not valid.


### .report
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/counterfactual_detection_results.py/#L153)
```python
.report()
```

---
Show an overview of the results.

Details:
For each concept, the maximum mean and maximum standard deviation of the counterfactual scores is shown.

### .dashboard
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/counterfactual_detection_results.py/#L164)
```python
.dashboard(
   num_keywords: int = 10, port: int = 8090
)
```

---
Start a dash dashboard with interactive box plots.


**Args**

* **num_keywords**  : The number of keywords per concept to show in the dashboard.
* **port**  : The port to run the dashboard on.


----


## CounterfactualConceptResult
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/counterfactual_detection_results.py/#L57)
```python 
CounterfactualConceptResult(
   concept: str, scores: pd.DataFrame, omitted_keywords: List[str],
   counterfactual_samples: Optional[List[CounterfactualSample]] = None
)
```


---
The result of a counterfactual bias detection run for a single concept.


**Attributes**

* **concept**  : The concept for which the result was calculated.
* **scores**  : The scores for the different keywords.
* **omitted_keywords**  : The keywords that were omitted from the analysis.
* **counterfactual_samples**  : The counterfactual samples that were generated.


----


## CounterfactualSample
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/counterfactual_detection_results.py/#L17)
```python 
CounterfactualSample(
   text: str, orig_keyword: str, keyword: str, concept: str,
   tokenized: TextRepresentation, score: Optional[float] = None,
   label: Optional[int|str] = None, source_text: Optional[str] = None
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
* **score**  : The counterfactual score of the sample.
* **label**  : The label of the original text.
* **source_text**  : The source text from which the text was derived.

