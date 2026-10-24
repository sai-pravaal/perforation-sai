"""Step 21: Outlier Thresholding.

File: backend/pipeline/phase4_statistics_output/step21_outlier_thresholding.py
Converts each hole's continuous combined_score into a yes/no outlier decision
and a tail probability (how often a genuine hole would score at least this high).

summed_z: the score is sqrt(sum of z^2) over the fields that contributed (d of
them). With ideal normal data and a large reference group, sum z^2 follows a
chi-square distribution with d degrees of freedom, and that is the cutoff the
spec intends. Real groups are small (4-25 holes per edge), so z from median/MAD
is heavy-tailed and the asymptotic chi-square cutoff over-flags badly (about
44% of genuine holes at the nominal 1% level for a 10-hole group). The null
distribution is therefore calibrated for the actual group size n and d by a
seeded Monte Carlo of exactly the same median/MAD z-score on normal data (a
standard Monte Carlo test). It converges to chi-square(d) as n grows.

mahalanobis: the squared distance is compared with chi-square(d) directly.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import chi2

from backend.config.pipeline_constants import (
    COMPOSITE_FIELD_ORDER,
    MAD_SCALE_CONSTANT,
    MIN_REFERENCE_GROUP_SIZE,
    NULL_CALIBRATION_DRAWS,
    NULL_CALIBRATION_SEED,
    OUTLIER_TAIL_PROBABILITY,
)
from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import HoleRecord
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)

_CHUNK = 2000


@lru_cache(maxsize=256)
def null_sum_z2(group_size: int, dof: int) -> np.ndarray:
    """Sorted Monte Carlo null distribution of sum(z^2) for a genuine hole.

    Each draw: a reference group of `group_size` and one test hole, all with
    `dof` independent standard-normal fields; z = 0.6745 (v - median) / MAD,
    exactly as Step 20 computes it. Seeded, so results are reproducible.
    """
    rng = np.random.default_rng(NULL_CALIBRATION_SEED + 1000 * group_size + dof)
    out = np.empty(NULL_CALIBRATION_DRAWS, dtype=np.float64)
    done = 0
    while done < NULL_CALIBRATION_DRAWS:
        m = min(_CHUNK, NULL_CALIBRATION_DRAWS - done)
        group = rng.standard_normal((m, group_size, dof))
        hole = rng.standard_normal((m, dof))
        med = np.median(group, axis=1)
        mad = np.median(np.abs(group - med[:, None, :]), axis=1)
        z = MAD_SCALE_CONSTANT * (hole - med) / np.maximum(mad, 1e-12)
        out[done:done + m] = np.sum(z * z, axis=1)
        done += m
    out.sort()
    return out


@lru_cache(maxsize=256)
def null_sum_z2_pooled(pool_sizes: Tuple[int, ...], dof: int, own_group_fields: int = 0) -> np.ndarray:
    """Sorted Monte Carlo null of sum(z^2) for Step 20's pooled-scale z.

    pool_sizes = (own-edge comparison count, other edge sizes...). Each draw
    rebuilds exactly Step 20's estimate on normal data: own-edge median centre,
    per-field MAD pooled over every edge's residuals, times the own-edge noise
    factor (median |own residual / pooled MAD|). own_group_fields of the dof
    fields (the circular shadow direction) are scored with the own group's MAD
    instead, as Step 20 does, and stay out of the noise factor.
    """
    own_n, others = int(pool_sizes[0]), [int(k) for k in pool_sizes[1:]]
    k_own = max(0, min(int(own_group_fields), dof))
    k_pool = dof - k_own
    rng = np.random.default_rng(NULL_CALIBRATION_SEED + 7919 * own_n + 31 * sum(others) + 3 * dof + k_own)
    out = np.empty(NULL_CALIBRATION_DRAWS, dtype=np.float64)
    done = 0
    while done < NULL_CALIBRATION_DRAWS:
        m = min(_CHUNK, NULL_CALIBRATION_DRAWS - done)
        own = rng.standard_normal((m, own_n, dof))
        hole = rng.standard_normal((m, dof))
        centre = np.median(own, axis=1)
        r_own = own - centre[:, None, :]
        total = np.zeros(m)
        if k_pool:
            parts = [r_own[:, :, :k_pool]]
            for k in others:
                g = rng.standard_normal((m, k, k_pool))
                parts.append(g - np.median(g, axis=1)[:, None, :])
            pooled = np.maximum(np.median(np.abs(np.concatenate(parts, axis=1)), axis=1), 1e-12)
            factor = np.median(np.abs(r_own[:, :, :k_pool] / pooled[:, None, :]).reshape(m, -1), axis=1)
            z = MAD_SCALE_CONSTANT * (hole[:, :k_pool] - centre[:, :k_pool]) / (
                pooled * np.maximum(factor, 1e-12)[:, None])
            total += np.sum(z * z, axis=1)
        if k_own:
            mad = np.maximum(np.median(np.abs(r_own[:, :, k_pool:]), axis=1), 1e-12)
            z = MAD_SCALE_CONSTANT * (hole[:, k_pool:] - centre[:, k_pool:]) / mad
            total += np.sum(z * z, axis=1)
        out[done:done + m] = total
        done += m
    out.sort()
    return out


def _null_for(group_size: int, dof: int, pool_sizes: Optional[Tuple[int, ...]],
              own_group_fields: int = 0) -> Optional[np.ndarray]:
    if pool_sizes and len(pool_sizes) > 1:
        if pool_sizes[0] < MIN_REFERENCE_GROUP_SIZE:
            return None
        return null_sum_z2_pooled(tuple(int(k) for k in pool_sizes), int(dof), int(own_group_fields))
    if group_size < MIN_REFERENCE_GROUP_SIZE:
        return None
    return null_sum_z2(int(group_size), int(dof))


def tail_probability(method: str, combined_score: float, group_size: int, dof: int,
                     pool_sizes: Optional[Tuple[int, ...]] = None, own_group_fields: int = 0) -> Optional[float]:
    """P(genuine hole scores >= combined_score); None when it cannot be calibrated."""
    if dof <= 0:
        return None
    s2 = float(combined_score) ** 2
    if method == "mahalanobis":
        return float(chi2.sf(s2, df=dof))
    null = _null_for(group_size, dof, pool_sizes, own_group_fields)
    if null is None:
        return None
    exceed = null.size - int(np.searchsorted(null, s2, side="left"))
    return (exceed + 1) / (null.size + 1)  # standard Monte Carlo p-value


def compute_threshold(method: str, dof: int, group_size: Optional[int] = None,
                      pool_sizes: Optional[Tuple[int, ...]] = None, own_group_fields: int = 0) -> float:
    """Cutoff on combined_score (a distance) at OUTLIER_TAIL_PROBABILITY.

    Without a group size (or for mahalanobis) this is the large-sample
    chi-square value; otherwise the calibrated Monte Carlo quantile for the
    group (or pooled edge structure) actually used.
    """
    dof = max(1, int(dof))
    if method == "mahalanobis" or group_size is None:
        return float(chi2.ppf(1.0 - OUTLIER_TAIL_PROBABILITY, df=dof) ** 0.5)
    null = _null_for(group_size, dof, pool_sizes, own_group_fields)
    if null is None:
        return float("inf")
    return float(np.quantile(null, 1.0 - OUTLIER_TAIL_PROBABILITY) ** 0.5)


def run_step21(
    session_id: str,
    holes: List[HoleRecord],
    deviation_scores: Dict[str, Dict[str, Any]],
    session_manager: Optional[SessionManager] = None
) -> Dict[str, Dict[str, Any]]:
    """Execute Step 21: Outlier Thresholding."""
    if session_manager is None:
        session_manager = SessionManager()

    session_dir = session_manager.get_session_dir(session_id)
    manifest = session_manager.get_manifest_manager(session_id)
    holes_dir = session_dir / "holes"

    flags: Dict[str, Dict[str, Any]] = {}

    for h in holes:
        score = deviation_scores.get(h.hole_id, {})
        method = score.get("method", "summed_z")
        combined = float(score.get("combined_score", 0.0))
        dof = int(score.get("degrees_of_freedom", len(COMPOSITE_FIELD_ORDER)))
        group_size = int(score.get("group_size", 0))
        pool = tuple(score.get("pool_sizes") or ()) if score.get("scale_method") == "pooled_edge_factor" else None
        k_own = int(score.get("own_group_fields", 0) or 0)
        threshold = compute_threshold(method, dof, None if method == "mahalanobis" else group_size, pool, k_own)
        p_value = tail_probability(method, combined, group_size, dof, pool, k_own)
        # A reference group too small to calibrate cannot support a flag.
        is_outlier = p_value is not None and p_value < OUTLIER_TAIL_PROBABILITY

        outlier = {
            "is_outlier": bool(is_outlier),
            "threshold_used": threshold if np.isfinite(threshold) else None,
            "tail_probability": p_value,
        }
        flags[h.hole_id] = outlier

        # Read-modify-write score.json so the "outlier" field is merged, not overwritten.
        score_path = holes_dir / h.hole_id / "score.json"
        data: Dict[str, Any] = {}
        if score_path.exists():
            try:
                with open(score_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = dict(score)
        data["outlier"] = outlier
        score_path.parent.mkdir(parents=True, exist_ok=True)
        with open(score_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # Feature 10 "flagged only" filter: tag every hole image with its outcome.
    manifest.set_image_flags({hid: f["is_outlier"] for hid, f in flags.items()})

    manifest.add_stage_entry(
        step_id=21,
        phase=4,
        stage_name="outlier_thresholding",
        output_path="holes/",
        status="complete"
    )

    return flags
