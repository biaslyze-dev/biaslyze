The main algorithm we use implemented in the [CounterfactualBiasDetector](../biaslyze/bias_detectors/counterfactual_biasdetector/), is an algorithm designed to detect bias by interchanging keywords associated with a protected concept and measuring the difference in outcomes for the model. The algorithm calculates a so called counterfactual score for each keyword, providing insights into potential biases in NLP ML applications. Here we outline the steps and processes involved in implementing the algorithm.

## Preparing the Dataset:
Before applying the bias detection algorithm, you need to gather a dataset that reflects the real-world context in which your NLP ML model operates. Ensure that the dataset includes a diverse representation of examples and accounts for different perspectives related to the protected concept you aim to investigate.

## Identifying Keywords:
Identify a set of keywords associated with the protected concept you want to examine for bias. These keywords should represent important terms or phrases that could influence the model's predictions or outcomes. For example, if you are analyzing bias related to gender, keywords might include "he," "she," "doctor," "nurse," and so on. We provide precompiled list of keywords for different concepts like gender or religion.

## Generating Counterfactual Examples:
For each keyword identified in the previous step, generate counterfactual examples by replacing the original keyword with a different keyword within the same category. This replacement helps simulate different scenarios and evaluate the model's sensitivity to changes in input while maintaining the context. For instance, if the original keyword is "he," you might replace it with "she" as a counterfactual example.

## Evaluating Model Outcomes:
Apply the NLP ML model to the original and counterfactual examples generated in the previous step. Record and compare the outcomes or predictions made by the model for each example. These outcomes could be classification labels, sentiment scores, or any other relevant outputs depending on the specific application.

## Calculating Counterfactual Scores:
Based on the differences observed in the model outcomes between the original and counterfactual examples, calculate a counterfactual score for each keyword. The counterfactual score represents the magnitude of the bias associated with a particular keyword. You can determine the score by measuring the degree of change in the model's outputs, such as the difference in predicted probabilities or the change in classification labels.

    +------------------------+               +------------------------+
    | Original   Text        |               | Counterfactual Text    |
    +------------------------+               +------------------------+
                   |                                    |
                   |                                    |
                   V                                    V
        +----------------+                     +------------------------+
        | NLP ML Model   |                     | NLP ML Model           |
        +----------------+                     +------------------------+
                   |                                    |
                   |                                    |
                   V                                    V
         +--------------+                       +----------------------+
         | Outcome      |                       | Outcome              |
         | (Original)   |                       | (Counterfactual)     |
         +--------------+                       +----------------------+
                   |                                    |
                   |                                    |
                   V                                    V
                   +------------------------------------+
                   |      Counterfactual Score          |
                   +------------------------------------+


## Analyzing Results:
Analyze the counterfactual scores obtained for each keyword to gain insights into potential bias in the NLP ML application. Higher counterfactual scores indicate that the model is more sensitive to changes in that keyword, suggesting a higher degree of bias associated with it. By identifying specific keywords with significant counterfactual scores, you can focus on addressing and mitigating potential biases in your model.

## Conclusion:
The algorithm described above provides a systematic approach for detecting bias in NLP ML applications. By interchanging keywords associated with a protected concept and measuring the resulting differences in model outcomes, the algorithm enables the calculation of counterfactual scores. These scores help identify specific keywords that may contribute to biases in the application. By integrating this algorithm into your bias detection pipeline and quality management process, you can enhance the fairness and inclusiveness of your NLP ML models.