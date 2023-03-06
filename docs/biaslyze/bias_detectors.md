#


## KeywordBiasDetector
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/bias_detectors.py/#L8)
```python 
KeywordBiasDetector(
   predict_func: Callable[[List[str]], List[float]],
   concept_detector = KeywordConceptDetector(),
   bias_evaluator = LimeBiasEvaluator()
)
```




**Methods:**


### .detect
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/bias_detectors.py/#L15)
```python
.detect(
   texts: List[str], labels: List
)
```

