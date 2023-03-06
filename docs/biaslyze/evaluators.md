#


## LimeBiasEvaluator
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluators.py/#L11)
```python 
LimeBiasEvaluator(
   n_lime_samples: int = 100
)
```




**Methods:**


### .evaluate
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluators.py/#L16)
```python
.evaluate(
   predict_func, texts: List[str], top_n: int = 10
)
```

---
Evaluate if a bias is present.
