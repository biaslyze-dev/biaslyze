#


## LimeDetectionResult
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/lime_detection_results.py/#L51)
```python 
LimeDetectionResult(
   biased_samples: List[LimeSampleResult]
)
```


---
Contains all samples on detected potential bias issues.


**Attributes**

* **biased_samples**  : A list of BiasedSampleResults.



**Methods:**


### .summary
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/lime_detection_results.py/#L62)
```python
.summary()
```

---
Print some summary statistics on the detected bias.

Answers the following questions:

- How many samples contain bias for the model?
- What concepts are affected?
- What are the reasons for the bias detection?

Example output:
```
Detected 1 samples with potential issues.
Potentially problematic concepts detected: Counter({'nationality': 1})
Based on keywords: Counter({'german': 1}).
---
```

### .details
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/lime_detection_results.py/#L80)
```python
.details(
   group_by_concept: bool = False
)
```

---
Print the details of every biased sample detected.


**Args**

* **group_by_concept**  : If the output should be grouped by concepts.


### .dashboard
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/lime_detection_results.py/#L117)
```python
.dashboard(
   use_position = False
)
```

---
Return a bokeh dashboard.

Content of the dashboard:
- A histogram of the LIME score of the samples.
- A table with the details of the samples.


**Args**

* **use_position**  : If True, use the normalized position for plotting.


----


## LimeSampleResult
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/results/lime_detection_results.py/#L9)
```python 
LimeSampleResult(
   text: str, bias_concepts: List[str], bias_reasons: List[str],
   top_words: List[str], num_tokens: List[str], keyword_position: int, score: float,
   metrics: Optional[Dict] = None
)
```


---
A sample on which the model might behave biased based on LIME.

Contains details on why it might be biased and what concepts are affected.


**Attributes**

* **text**  : Sample text
* **bias_concepts**  : Protected concepts present in the text by which the model appears biased.
* **bias_reasons**  : Reasons why bias was detected. Might be a list of keywords.
* **top_words**  : Most important words for the prediction.
* **num_tokens**  : Number of unique tokens in the text.
* **keyword_position**  : Position of the keyword in the top_words list.
* **score**  : Score of the sample.
* **metrics**  : Metrics of the LIME explainer.

