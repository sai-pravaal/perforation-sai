"""Step 20: Robust Statistics & Combined Deviation Score.

File: backend/pipeline/phase4_statistics_output/step20_robust_stats_deviation_score.py
For each composite field, computes the median and MAD across a hole's comparison
group (never itself), converts the hole's own value to a robust z-score, and
combines them into one deviation score via summed-z (L2 norm, sqrt of the sum of
z squared) or Mahalanobis distance, per COMBINATION_METHOD.

shadow_direction_deg is an angle, so it is scored with circular statistics:
circular median centre, wrapped deviations in (-180, 180], MAD of those
deviations, and directions from weak shadows left out (STAT-02).

Each score records the reference group size and degrees of freedom (fields that
actually contributed) so Step 21 can calibrate the cutoff for that group.
"""

import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import median_abs_deviation

from backend.config.pipeline_constants import (
    COMBINATION_METHOD,
    COMPOSITE_FIELD_ORDER,
    MAD_SCALE_CONSTANT,
    MEAN_AD_SCALE_CONSTANT,
    POOLED_SCALE_MIN_EDGE_SIZE,
    SCALE_METHOD,
    SHADOW_DIRECTION_MIN_RELATIVE_STRENGTH,
    SHADOW_DIRECTION_MIN_RESULTANT_LENGTH,
    ZERO_SPREAD_RELATIVE_FLOOR,
)
from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import HoleRecord
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)

_EPS = 1e-9
_DIR_IDX = COMPOSITE_FIELD_ORDER.index("shadow_direction_deg")
_STRENGTH_IDX = COMPOSITE_FIELD_ORDER.index("shadow_strength")


def wrap_degrees(delta: np.ndarray) -> np.ndarray:
    """Fold angle differences into (-180, 180]."""
    return -((-np.asarray(delta, dtype=np.float64) + 180.0) % 360.0 - 180.0)


def circular_median_deg(angles_deg: np.ndarray) -> float:
    """Circular (quasi-)median: the sample angle minimising the summed absolute wrapped deviation."""
    a = np.asarray(angles_deg, dtype=np.float64)
    costs = [float(np.sum(np.abs(wrap_degrees(a - c)))) for c in a]
    return float(a[int(np.argmin(costs))] % 360.0)


def _circular_direction_z(
    v_dir: float, v_strength: float, dirs: np.ndarray, strengths: np.ndarray
) -> Tuple[float, float, float, bool]:
    """Robust z for the shadow direction. Returns (z, centre_deg, mad_deg, contributed).

    - Directions from shadows weaker than SHADOW_DIRECTION_MIN_RELATIVE_STRENGTH x
      the group's median strength are left out (their angle is mostly noise).
    - If the remaining directions have mean resultant length below
      SHADOW_DIRECTION_MIN_RESULTANT_LENGTH there is no common direction to
      deviate from, so the field is undefined for this group (not scored).
    - Otherwise: circular median centre, wrapped deviations, MAD of the
      deviations, z = 0.6745 x wrapped(hole - centre) / MAD.
    """
    ok = np.isfinite(dirs) & np.isfinite(strengths)
    if not np.isfinite(v_dir) or not np.any(ok):
        return 0.0, float("nan"), float("nan"), False
    median_strength = float(np.median(strengths[ok]))
    floor = SHADOW_DIRECTION_MIN_RELATIVE_STRENGTH * median_strength
    keep = ok & (strengths >= floor)
    if np.count_nonzero(keep) < 2:
        return 0.0, float("nan"), float("nan"), False
    theta = np.radians(dirs[keep])
    resultant = math.hypot(float(np.mean(np.cos(theta))), float(np.mean(np.sin(theta))))
    if resultant < SHADOW_DIRECTION_MIN_RESULTANT_LENGTH:
        return 0.0, float("nan"), float("nan"), False
    centre = circular_median_deg(dirs[keep])
    deviations = wrap_degrees(dirs[keep] - centre)
    mad = float(np.median(np.abs(deviations)))
    if np.isfinite(v_strength) and v_strength < floor:
        # This hole's own shadow is too weak for its direction to mean anything.
        return 0.0, centre, mad, False
    hole_dev = float(wrap_degrees(v_dir - centre))
    if mad >= _EPS:
        z = MAD_SCALE_CONSTANT * hole_dev / mad
    else:
        z = _zero_mad_z(hole_dev, 0.0, deviations)
    return z, centre, mad, True


def robust_z_details(value_vec: np.ndarray, group_matrix: np.ndarray):
    """Per-field robust z, group centre and MAD, and a mask of fields that contributed.

    group_matrix: shape (n_group, n_fields). NaN-safe (roundness is NaN for teeth).
    """
    n_fields = value_vec.shape[0]
    # An all-NaN field (e.g. roundness across a group of teeth) is expected; the
    # resulting NaN median/MAD is handled below as "no contribution".
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        median = np.nanmedian(group_matrix, axis=0)
        mad = median_abs_deviation(group_matrix, axis=0, nan_policy="omit", scale=1.0)

    z = np.zeros(n_fields, dtype=np.float64)
    used = np.zeros(n_fields, dtype=bool)
    circular = n_fields == len(COMPOSITE_FIELD_ORDER)
    for i in range(n_fields):
        if circular and i == _DIR_IDX:
            z[i], median[i], mad[i], used[i] = _circular_direction_z(
                value_vec[i], value_vec[_STRENGTH_IDX], group_matrix[:, i], group_matrix[:, _STRENGTH_IDX]
            )
            continue
        v = value_vec[i]
        m = median[i]
        d = mad[i]
        if not np.isfinite(v) or not np.isfinite(m) or not np.isfinite(d):
            continue  # no data for this field (e.g. roundness across teeth)
        used[i] = True
        if d >= _EPS:
            z[i] = MAD_SCALE_CONSTANT * (v - m) / d
        else:
            z[i] = _zero_mad_z(v, m, group_matrix[:, i])
    return z, median, mad, used


def pooled_scale_z_details(
    value_vec: np.ndarray, own_matrix: np.ndarray, other_edge_matrices: List[np.ndarray]
):
    """Robust z with the spread borrowed across the specimen's edges (STAT-01 follow-up).

    Centre: median of the hole's own-edge comparison group (leave-one-out), as before.
    Spread for field f: pooled MAD of residuals about each edge's own median,
    taken over the own-edge group and every other edge-grouped edge of this
    specimen, multiplied by an edge noise factor = median |own-edge residual /
    pooled MAD| across all fields. The pooled MAD borrows strength across edges
    (empirical-Bayes idea, cf. limma, Smyth 2004); the factor keeps a genuinely
    noisier edge (e.g. compound perforation) from being over-flagged.
    z = 0.6745 (v - centre) / (pooled MAD x factor), so a homogeneous specimen
    gives the same scale as the per-edge formula, just estimated from more holes.

    Returns (z, centre, scale_mad, used, factor). shadow_direction_deg keeps its
    circular per-group treatment.
    """
    n_fields = value_vec.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        centre = np.nanmedian(own_matrix, axis=0)
        residuals = [own_matrix - centre]
        for m in other_edge_matrices:
            residuals.append(m - np.nanmedian(m, axis=0))
        stacked = np.vstack(residuals)
        pooled_mad = np.nanmedian(np.abs(stacked), axis=0)
        ratios = np.abs(residuals[0]) / np.where(pooled_mad >= _EPS, pooled_mad, np.nan)
        if n_fields == len(COMPOSITE_FIELD_ORDER):
            ratios[:, _DIR_IDX] = np.nan  # the angle is not part of the linear pool
        finite_ratios = ratios[np.isfinite(ratios)]
    factor = float(np.median(finite_ratios)) if finite_ratios.size else 1.0
    if not np.isfinite(factor) or factor < _EPS:
        factor = 1.0

    z = np.zeros(n_fields, dtype=np.float64)
    used = np.zeros(n_fields, dtype=bool)
    scale = pooled_mad * factor
    circular = n_fields == len(COMPOSITE_FIELD_ORDER)
    for i in range(n_fields):
        if circular and i == _DIR_IDX:
            z[i], centre[i], scale[i], used[i] = _circular_direction_z(
                value_vec[i], value_vec[_STRENGTH_IDX], own_matrix[:, i], own_matrix[:, _STRENGTH_IDX]
            )
            continue
        v, m, d = value_vec[i], centre[i], scale[i]
        if not np.isfinite(v) or not np.isfinite(m) or not np.isfinite(d):
            continue
        used[i] = True
        if d >= _EPS:
            z[i] = MAD_SCALE_CONSTANT * (v - m) / d
        else:
            z[i] = _zero_mad_z(v - m, 0.0, stacked[:, i])
    return z, centre, scale, used, factor


def robust_z_scores(value_vec: np.ndarray, group_matrix: np.ndarray):
    """Per-field robust z, plus group median and MAD vectors (see robust_z_details)."""
    z, median, mad, _ = robust_z_details(value_vec, group_matrix)
    return z, median, mad


def _zero_mad_z(v: float, m: float, column: np.ndarray) -> float:
    """Robust z when the group MAD is zero (more than half the group shares one value).

    Returning 0 here would hide a hole that differs sharply from an otherwise
    identical group, so the spread falls back to the mean absolute deviation
    about the median (scaled to a normal-consistent sigma), and if that is also
    zero, to a small fraction of |median|. A hole equal to the median still gets 0.
    """
    if v == m:
        return 0.0
    col = column[np.isfinite(column)]
    mean_ad = float(np.mean(np.abs(col - m))) if col.size else 0.0
    if mean_ad >= _EPS:
        logger.debug("Step 20: MAD=0, using mean absolute deviation fallback")
        return MEAN_AD_SCALE_CONSTANT * (v - m) / mean_ad
    floor = max(ZERO_SPREAD_RELATIVE_FLOOR * abs(m), _EPS)
    logger.debug("Step 20: group has zero spread, using relative floor %.3g", floor)
    return (v - m) / floor


def _mahalanobis_distance(value_vec: np.ndarray, group_matrix: np.ndarray) -> Tuple[float, int]:
    """Robust Mahalanobis distance using scikit-learn's MinCovDet (lazy import).

    Returns (distance, number of fields used). The circular shadow direction is
    excluded: a covariance of raw angles is meaningless across the 0/360 wrap.
    """
    from sklearn.covariance import MinCovDet

    # Use only fields measured for this hole and for every group member; an
    # all-NaN column (roundness across teeth) zero-filled would be a constant
    # column and make the covariance singular.
    usable = np.isfinite(value_vec) & np.all(np.isfinite(group_matrix), axis=0)
    if value_vec.shape[0] == len(COMPOSITE_FIELD_ORDER):
        usable[_DIR_IDX] = False
    if not np.any(usable) or group_matrix.shape[0] <= int(np.sum(usable)):
        raise ValueError("not enough complete fields/members for a robust covariance fit")
    clean = group_matrix[:, usable]
    val = value_vec[usable]
    mcd = MinCovDet().fit(clean)
    dist_sq = float(mcd.mahalanobis(val.reshape(1, -1))[0])
    return math.sqrt(max(0.0, dist_sq)), int(np.sum(usable))


def run_step20(
    session_id: str,
    holes: List[HoleRecord],
    composites: Dict[str, Dict[str, Any]],
    reference_groups: Dict[str, Dict[str, Any]],
    session_manager: Optional[SessionManager] = None
) -> Dict[str, Dict[str, Any]]:
    """Execute Step 20: Robust Statistics & Combined Deviation Score."""
    if session_manager is None:
        session_manager = SessionManager()

    session_dir = session_manager.get_session_dir(session_id)
    manifest = session_manager.get_manifest_manager(session_id)
    holes_dir = session_dir / "holes"

    scores: Dict[str, Dict[str, Any]] = {}

    # Teeth on edges large enough for edge-level comparison share one pooled
    # spread estimate (SCALE_METHOD "pooled_edge_factor"). Pooled-fallback teeth
    # and interior holes have a single group, so they keep the group MAD.
    def _vec(hid: str) -> List[float]:
        return [np.nan if v is None else v for v in composites[hid]["vector"]]

    edge_members: Dict[str, List[str]] = {}
    if SCALE_METHOD == "pooled_edge_factor":
        for h in holes:
            group = reference_groups.get(h.hole_id, {})
            if group.get("stratification") == "edge" and h.hole_id in composites and composites[h.hole_id].get("vector"):
                edge_members.setdefault(h.edge, []).append(h.hole_id)

    for h in holes:
        comp = composites.get(h.hole_id, {})
        vector = np.asarray([np.nan if v is None else v for v in comp.get("vector", [])], dtype=np.float64)
        group_ids = reference_groups.get(h.hole_id, {}).get("comparison_group_ids", [])
        group_vectors = [
            [np.nan if v is None else v for v in composites[gid]["vector"]] for gid in group_ids
            if gid in composites and composites[gid].get("vector")
        ]

        # method_used records what actually produced combined_score, so Step 21
        # picks the matching cutoff even when Mahalanobis falls back.
        method_used = "summed_z"
        dof = 0
        scale_method = "group_mad"
        pool_sizes: List[int] = []
        edge_factor: Optional[float] = None
        if vector.size == 0 or len(group_vectors) == 0:
            per_field_z = [0.0] * int(vector.size)
            combined = 0.0
            median = [None] * int(vector.size)
            mad = [None] * int(vector.size)
        else:
            group_matrix = np.asarray(group_vectors, dtype=np.float64)
            other_edges = [
                np.asarray([_vec(hid) for hid in ids], dtype=np.float64)
                for edge, ids in edge_members.items() if edge != h.edge
            ]
            own_edge = edge_members.get(h.edge, [])
            if h.hole_id in own_edge and len(own_edge) >= POOLED_SCALE_MIN_EDGE_SIZE and other_edges:
                z, median_arr, mad_arr, used, edge_factor = pooled_scale_z_details(
                    vector, group_matrix, other_edges)
                scale_method = "pooled_edge_factor"
                pool_sizes = [len(group_vectors)] + [m.shape[0] for m in other_edges]
            else:
                z, median_arr, mad_arr, used = robust_z_details(vector, group_matrix)
            per_field_z = [float(x) for x in z]
            dof = int(np.sum(used))
            # A field with no usable group data has no centre/MAD: store null
            # (NaN is not valid JSON).
            median = [float(x) if np.isfinite(x) else None for x in median_arr]
            mad = [float(x) if np.isfinite(x) else None for x in mad_arr]

            combined = float(math.sqrt(np.sum(z[used] ** 2)))
            if COMBINATION_METHOD == "mahalanobis":
                try:
                    combined, dof = _mahalanobis_distance(vector, group_matrix)
                    method_used = "mahalanobis"
                except Exception as exc:
                    logger.warning("Mahalanobis unavailable for %s (%s); using summed_z", h.hole_id, exc)

        score = {
            "per_field_z": per_field_z,
            "combined_score": float(combined),
            "method": method_used,
            "group_median": median,
            "group_mad": mad,
            "group_size": len(group_vectors),
            "degrees_of_freedom": dof,
            # How the spread was estimated, and the edge sizes Step 21 needs to
            # calibrate the matching null distribution.
            "scale_method": scale_method,
            "pool_sizes": pool_sizes,
            "edge_noise_factor": edge_factor,
            # Fields scored with the own group's spread inside a pooled score
            # (the circular shadow direction), for Step 21's matching null.
            "own_group_fields": int(scale_method == "pooled_edge_factor" and dof > 0
                                    and vector.size == len(COMPOSITE_FIELD_ORDER)
                                    and bool(used[_DIR_IDX])) if len(group_vectors) else 0,
        }
        scores[h.hole_id] = score

        h_dir = holes_dir / h.hole_id
        h_dir.mkdir(parents=True, exist_ok=True)
        with open(h_dir / "score.json", "w", encoding="utf-8") as f:
            json.dump(score, f, indent=2)

    manifest.add_stage_entry(
        step_id=20,
        phase=4,
        stage_name="robust_stats_deviation_score",
        output_path="holes/",
        status="complete"
    )

    return scores
