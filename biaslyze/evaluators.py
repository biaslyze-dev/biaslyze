"""Contains classes to evaluate the bias of detected concepts."""
from typing import List
import numpy as np
from tqdm import tqdm
import warnings
from eli5.lime import TextExplainer

from concepts import CONCEPTS


class LimeBiasEvaluator:
    def __init__(self):
        self.explainer = TextExplainer(n_samples=100)

    def evaluate(self, predict_func, texts: List[str], labels: List, top_n: int = 10):
        """Evaluate if a bias is present."""
        warnings.filterwarnings("ignore", category=FutureWarning)
        biased_samples = []
        for text in tqdm(texts):
            self.explainer.fit(text, predict_func)
            interpret_sample_dict = {
                coef: token
                for coef, token in zip(
                    self.explainer.clf_.coef_[0],
                    self.explainer.vec_.get_feature_names_out(),
                )
            }
            top_interpret_sample_dict = sorted(
                interpret_sample_dict.items(), key=lambda x: -np.abs(x[0])
            )[: min(len(interpret_sample_dict), top_n)]
            important_tokens = [w.lower() for (_, w) in top_interpret_sample_dict]
            if (
                len(
                    set(CONCEPTS.get("nationality")).intersection(set(important_tokens))
                )
                > 0
            ):
                biased_samples.append((text, important_tokens))

        return biased_samples
