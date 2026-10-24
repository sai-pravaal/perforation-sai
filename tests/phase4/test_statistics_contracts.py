import numpy as np

from backend.pipeline.phase4_statistics_output.step20_robust_stats_deviation_score import robust_z_scores
from backend.pipeline.phase4_statistics_output.step21_outlier_thresholding import compute_threshold


def test_statistics_and_threshold_contracts():
    scores, median, mad = robust_z_scores(np.array([2.0]), np.array([[1.0], [1.0], [1.0]]))
    assert scores.shape == median.shape == mad.shape == (1,)
    # A zero-MAD group must not hide a differing hole (STAT-03).
    assert scores[0] > 0.0
    same, _, _ = robust_z_scores(np.array([1.0]), np.array([[1.0], [1.0], [1.0]]))
    assert same[0] == 0.0
    # Large-sample reference: chi-square(14) at the 1% tail on the L2 norm (~5.40).
    assert abs(compute_threshold("summed_z", 14) - 5.398) < 0.01
    assert compute_threshold("mahalanobis", 14) > 0.0
    # Small groups get a calibrated (larger) cutoff, converging to the reference.
    assert compute_threshold("summed_z", 14, group_size=10) > compute_threshold("summed_z", 14, group_size=200) > 5.4
