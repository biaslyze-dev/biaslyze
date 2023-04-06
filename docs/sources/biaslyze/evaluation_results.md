#


## EvaluationResult
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluation_results.py/#L27)
```python 
EvaluationResult(
   biased_samples: List[BiasedSampleResult]
)
```


---
Contains all samples on detected potential bias issues.

Attribues:
biased_samples: A list of BiasedSampleResults.


**Methods:**


### .summary
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluation_results.py/#L37)
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
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluation_results.py/#L55)
```python
.details(
   group_by_concept: bool = False
)
```

---
Print the details of every biased sample detected.


**Args**

* **group_by_concept**  : If the output should be grouped by concepts.


----


## BiasedSampleResult
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluation_results.py/#L7)
```python 
BiasedSampleResult(
   text: str, bias_concepts: List[str], bias_reasons: List[str]
)
```


---
A sample on which the model might behave biased.

Contains details on why it might be biased and what concepts are affected.


**Attributes**

* **text**  : Sample text
* **bias_concepts**  : Protected concepts present in the text by which the model appears biased.
* **bias_reasons**  : Reasons why bias was detected. Might be a list of keywords.

