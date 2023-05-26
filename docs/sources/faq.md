# FAQ

## Why use Biaslyze?
Bias is often subtle and difficult to detect in NLP models, as the protected attributes are less obvious and can take many forms in language (e.g. proxies, double meanings, ambiguities etc.). 
Therefore, technical bias testing is a key step in avoiding algorithmically mediated discrimination. However, it is currently conducted too rarely due to the effort involved, missing resources or lack of awareness for the problem.

Biaslyze helps to get started with the analysis of bias within NLP models and offers a concrete entry point for further impact assessments and mitigation measures. 
Especially for less experienced developers, students and teams with limited resources, our toolbox offers a low-effort approach to bias testing in NLP use cases.


## Who can use Biaslyze? 
Biaslyze is primarily designed for developers of software containing NLP components. 
In general, however, Biaslyze can be used by anyone who is interested in an ethically responsible use of text-based machine learning application. What you need is a basic understanding of programming in Python and NLP applications.

## What Biaslyze can not do
Biaslyze will not make any statements about whether a model is biased or not. It will neither tell you if the importance of an attribute or concept within your model is problematic ot not. The answer to this question is complex and depends, on the application context, the embedding and many other factors that can not be assessed by our toolbox. Biaslyze will not make your model or application bias-free or discrimination free.


## What Biaslyze can do
Biaslyze can evaluate the importance of certain sensitive attributes in a model and show you indications for bias. The attributes which are in many cases under the protection of anti-discrimination laws are among others gender, nationality and religion. Biaslyze can help you notice and start assessing or possibly mitigating sources of bias.
Please also look at the [tutorial](tutorials/tutorial-toxic-comments/) to get a sense of how Biaslyze works.


## Purpose
The tool was built for NLP applications and should therefore also be used for this area of application.

## Languages
Biaslyze currently works for English and operates with English keywords. However, it is our wish to provide Biaslyze for German and potentially other languages in the future. 
We hope to expand the language selection soon.

## Counterfactuals and Keyword Lists
We are aware that the method of counterfactual token fairness has its weaknesses. We therefore also want to point out that the keyword lists compiled in the used concepts are not exhaustive. 
Our aim is to expand the concepts and keywords accordingly, involving a multitude of perspectives and individuals.


