#


## MaskedKeywordBiasDetector
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/bias_detectors/masked_keyword_biasdetector.py/#L11)
```python 
MaskedKeywordBiasDetector(
   predict_func: Callable[[List[str]], List[float]], n_resample_keywords: int = 10,
   concept_detector = KeywordConceptDetector(),
   bias_evaluator = MaskedLMBiasEvaluator()
)
```


---
Detect bias by finding keywords related to protected concepts with a language model that change the prediction a lot.


**Methods:**


### .detect
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/bias_detectors/masked_keyword_biasdetector.py/#L26)
```python
.detect(
   texts: List[str]
)
```

