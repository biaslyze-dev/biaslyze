"""Contains several utility functions for the biaslyze package."""
import socket

import dill


def load_results(path: str):
    """Load saved CounterfactualDetectionResults from a file.

    For previously with CounterfactualBiasDetector.save() saved results.
    """
    with open(path, "rb") as f:
        return dill.load(f)


def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use.

    Args:
        port: The port to check.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
        return False  # Port is available
    except OSError:
        return True  # Port is already in use
