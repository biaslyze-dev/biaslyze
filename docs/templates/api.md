# API Reference

Welcome to the API reference for the biaslyze package. This document provides an overview of the key components and classes available in the package's API.

## Bias Detectors: CounterfactualBiasDetector

The `CounterfactualBiasDetector` class is at the heart of the biaslyze package, enabling users to identify biases in NLP models by employing counterfactual analysis. This detector examines text input and generates counterfactual examples to highlight potential bias of the model. 

## Results: CounterfactualDetectionResult

After running bias detection using the `CounterfactualBiasDetector`, the `CounterfactualDetectionResult` class is used to capture and present the results. This class provides comprehensive information about the detected biases, such as the identified biased concepts, the generated counterfactual examples, and most importantly, visualizations of the results. This facilitates a deeper understanding of the biases present in the analyzed text.

## Concepts

The `Concept` class defines the fundamental building block for bias analysis in the biaslyze package. Concepts represent specific terms, phrases, or ideas related to protected attributes that might be affected by bias. These can be predefined concepts, or users can define their own based on the context of their analysis.


## Concept Detectors

The `KeywordConceptDetector` enables the package to identify the presence of concepts in textual data for further downstream analysis.

## TextRepresentation

The `TextRepresentation` class assists in creating structured representations of textual content. It is a crucial component for bias detection, enabling the `CounterfactualBiasDetector` to generate counterfactual examples that emphasize potential biases. This class aids in converting raw text into a format suitable for bias analysis.

## Utils

The Utils module contains utility functions that support various operations within the biaslyze package. 

With these essential components and classes, the biaslyze package provides a comprehensive framework for detecting, analyzing, and understanding biases within NLP models. Whether you are working on social media content, news articles, or any other form of text, the biaslyze package equips you with the tools needed to uncover potential biases and make more informed decisions.
