---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Test biaslyze with disaster tweets data

Data source: https://www.kaggle.com/competitions/nlp-getting-started/overview

```python
%load_ext autoreload
%autoreload 2
```

```python
import sys
sys.path.append('/home/tobias/Repositories/biaslyze/')
```

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
```

## Load and prepare data

```python
df = pd.read_csv("../data/disaster-tweets/train.csv"); df.head()
```

```python
# replace urls
import re
url_regex = re.compile("(http|https)://[\w\-]+(\.[\w\-]+)+\S*")

df = df.replace(to_replace=url_regex, value='', regex=True)
```

## Train a model

```python
clf = make_pipeline(TfidfVectorizer(min_df=10, max_features=10000, stop_words="english"), LogisticRegression(n_jobs=4))
```

```python
clf.fit(df.text, df.target)
```

```python
train_pred = clf.predict(df.text)
print(accuracy_score(df.target, train_pred))
```

## Test detection of concepts

```python
from biaslyze.concept_detectors import KeywordConceptDetector
from biaslyze.evaluators import LimeBiasEvaluator
```

```python
key_detect = KeywordConceptDetector()
```

```python
detected_tweets = key_detect.detect(texts=df.text[:600])
```

```python
len(detected_tweets)
```

```python
detected_tweets
```

## Test LIME based bias detection with keywords

```python
from biaslyze.bias_detectors import LimeKeywordBiasDetector
```

```python
bias_detector = LimeKeywordBiasDetector(bias_evaluator=LimeBiasEvaluator(n_lime_samples=5000), n_top_keywords=5, use_tokenizer=True)
```

```python
detection_res = bias_detector.detect(texts=df.text.sample(frac=0.1), predict_func=clf.predict_proba)
```

```python
detection_res.summary()
```

```python
detection_res.details(group_by_concept=True)
```

## Testing a sentiment analysis model from huggingface

```python
from transformers import pipeline
from torch.utils.data import Dataset


classifier = pipeline(
    model="distilbert-base-uncased-finetuned-sst-2-english",
    top_k=None,
    padding=True,
    truncation=True
)
```

```python
class MyDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def predict_sentiment(texts):
    data = MyDataset(texts)
    proba = []
    for res in classifier(data):
        proba_array = []
        for p in sorted(res, key=lambda d: d['label'], reverse=True):
            proba_array.append(p.get("score"))
        proba.append(np.array(proba_array))
    return np.array(proba) / np.array(proba).sum(axis=1)[:,None]
```

```python
bias_detector = LimeKeywordBiasDetector(
    bias_evaluator=LimeBiasEvaluator(n_lime_samples=500),
    n_top_keywords=10,
    use_tokenizer=True
)
```

```python
test_texts = detected_tweets[:10]
detection_res = bias_detector.detect(texts=test_texts, predict_func=predict_sentiment)
```

```python
detection_res.summary()
```

```python
detection_res.details(group_by_concept=True)
```

## !! Very Experimental !!: Test masked language model based bias detection with keywords

```python
from biaslyze.bias_detectors import MaskedKeywordBiasDetector
```

```python
bias_detector = MaskedKeywordBiasDetector(n_resample_keywords=15, use_tokenizer=True)
```

```python
detection_res = bias_detector.detect(texts=df.text[500:600], predict_func=predict_sentiment)
```

```python
detection_res.summary()
```

```python
detection_res.details()
```

```python

```
