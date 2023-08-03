"""Contains several utility functions for the biaslyze package."""
import dill


def load_results(path: str):
    """Load saved CounterfactualDetectionResults from a file.
    
    For previously with CounterfactualBiasDetector.save() saved results.
    """
    with open(path, "rb") as f:
        return dill.load(f)