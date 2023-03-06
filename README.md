# biaslyze
The NLP Bias Identification Toolkit


## Usage example

```python
from biaslyze.bias_detectors import LimeKeywordBiasDetector

bias_detector = LimeKeywordBiasDetector(
    predict_func=clf.predict_proba,    # here, clf is a scikit-learn text classification pipeline trained for a binary classification task
    bias_evaluator=LimeBiasEvaluator(n_lime_samples=500),
    n_top_keywords=10
)

# detect bias in the model based on the given texts
detection_res = bias_detector.detect(texts)

# see a summary of the detection
detection_res.summary()
```

## Development setup


## Contributing

Follow the google style guide for python: https://google.github.io/styleguide/pyguide.html

