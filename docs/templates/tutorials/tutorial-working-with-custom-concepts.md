# Tutorial: How to work with custom concepts

While the biaslyze package contains a number of concepts for both English and German, you might want to work with your own concepts. This tutorial will show you how to do that.

First we need to import the `Concepts` class from the `biaslyze` package:

```python
from biaslyze.concepts import Concept
```

## Define a custom concept

Now, we can define a new concept. Let's say we want to detect the effect of names in German texts on the prediction of our model. We can define a new concept by passing a List of Dictionaries, where each Dictionary contains a keyword with some metadata.

```python
names_concept = Concept.from_dict_keyword_list(
    name="names",
    lang="de",
    keywords=[{"keyword": "Hans", "function": ["name"]}],
)
```

## Use the custom concept in the `CounterfactualBiasDetector`

When this is done, we can register the concept with the `CounterfactualBiasDetector` class. This will make sure that the concept is used when we call the `process` method of the `CounterfactualBiasDetector` class.

```python
bias_detector = CounterfactualBiasDetector(lang="de")
bias_detector.register_concept(names_concept)
```

Now we can finally use the concept by calling the `process` method of the `CounterfactualBiasDetector` class. We will use the `predict_proba` method of the `clf` object to predict the probabilities of the given texts.

```python
detection_res = bias_detector.process(
    texts=texts,
    predict_func=clf.predict_proba
    concepts_to_consider=["names"],
)
```