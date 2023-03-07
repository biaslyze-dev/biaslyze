#


## LimeBiasEvaluator
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluators.py/#L16)
```python 
LimeBiasEvaluator(
   n_lime_samples: int = 100
)
```


---
Evaluate bias in text based on LIME.


**Attributes**

* **n_lime_samples**  : Number of perturbed samples to create for each LIME run.



**Methods:**


### .evaluate
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluators.py/#L27)
```python
.evaluate(
   predict_func, texts: List[str], top_n: int = 10
)
```

---
Evaluate if a bias is present with LIME.


**Args**

* **predict_func**  : Function that predicts a for a given text. Currently only binary classification is supported.
* **texts**  : List of texts to evaluate.
* **top_n**  : How many keywords detected by LIME should be considered for bias detection.


**Returns**

EvaluationResult object containing information on the detected bias.

----


## MaskedLMBiasEvaluator
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluators.py/#L79)
```python 

```




**Methods:**


### .evaluate
[source](https://github.com/biaslyze-dev/biaslyze/blob/main/biaslyze/evaluators.py/#L85)
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
