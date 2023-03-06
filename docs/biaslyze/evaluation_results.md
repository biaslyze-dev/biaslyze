#


## EvaluationResult
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluation_results.py/#L16)
```python 
EvaluationResult(
   biased_samples: List[BiasedSampleResult]
)
```




**Methods:**


### .summary
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluation_results.py/#L20)
```python
.summary()
```


### .details
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluation_results.py/#L23)
```python
.details()
```

---
Print the details of every biased sample detected.

----


## BiasedSampleResult
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluation_results.py/#L6)
```python 
BiasedSampleResult(
   text: str, bias_concepts: List[str], bias_reasons: List[str]
)
```


