#


## EvaluationResult
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluation_results.py/#L15)
```python 
EvaluationResult(
   biased_samples: List[BiasedSampleResult]
)
```




**Methods:**


### .summary
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluation_results.py/#L19)
```python
.summary()
```


### .details
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluation_results.py/#L22)
```python
.details()
```

---
Print the details of every biased sample detected.

----


## BiasedSampleResult
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluation_results.py/#L5)
```python 
BiasedSampleResult(
   text: str, bias_concepts: List[str], bias_reasons: List[str]
)
```


