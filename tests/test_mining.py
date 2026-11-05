import numpy as np
import pandas as pd

from src.clustering import cluster
from src.log_preprocessor import preprocess


def test_preprocessor_removes_blank_short_and_duplicate_logs():
    frame = pd.DataFrame({"consumer_complaint_narrative": ["", "too short", "A meaningful payment exception narrative " * 2, "A meaningful payment exception narrative " * 2]})
    result = preprocess(frame)
    assert len(result) == 1


def test_clustering_returns_one_label_per_vector():
    labels, projection = cluster(np.random.default_rng(42).normal(size=(12, 4)), n_clusters=3)
    assert len(labels) == 12
    assert projection.shape == (12, 2)
