#


## KeywordBiasDetector
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/bias_detectors.py/#L8)
```python 
KeywordBiasDetector(
   predict_func: Callable[[List[str]], List[float]], n_top_keywords: int = 10,
   concept_detector = KeywordConceptDetector(),
   bias_evaluator = LimeBiasEvaluator()
)
```




**Methods:**


### .detect
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/bias_detectors.py/#L21)
```python
.detect(
   texts: List[str]
)
```

