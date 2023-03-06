#


## LimeBiasEvaluator
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluators.py/#L14)
```python 
LimeBiasEvaluator(
   n_lime_samples: int = 100
)
```




**Methods:**


### .evaluate
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluators.py/#L19)
```python
.evaluate(
   predict_func, texts: List[str], top_n: int = 10
)
```

---
Evaluate if a bias is present with LIME.

----


## MaskedLMBiasEvaluator
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluators.py/#L58)
```python 

```




**Methods:**


### .evaluate
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluators.py/#L62)
```python
.evaluate(
   predict_func, texts: List[str], n_resample_keywords: int = 10
)
```

---
Evaluate if a bias is present with masked language models.

Use a masked language model to resample keywords in texts and measure the difference in prediction.
If the difference is 'large' we call is biased sample.


**Note**

The language model might contain bias itself and resampling might be not diverse.
