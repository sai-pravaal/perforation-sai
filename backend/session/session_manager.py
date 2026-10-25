"""IPDS Session Manager.

Manages session filesystem lifecycle under data/sessions/<session_id>/.
Creates directories (raw, stages, holes, results), discovers incomplete
sessions for crash-resume, handles session resets, and persists SessionLog.
"""

import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config.capture_schema import CAPTURE_SCHEMA, match_filename_to_tag
from backend.session.manifest import ManifestManager


class SessionManager:
    """Manages session creation, directory structures, and session log I/O."""

    def __init__(self, sessions_root: Optional[Path | str] = None) -> None:
        if sessions_root is None:
            # Default to <workspace_root>/data/sessions
            self.sessions_root = Path(__file__).resolve().parents[2] / "data" / "sessions"
        else:
            self.sessions_root = Path(sessions_root)
        self.sessions_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_session_id(session_id: str) -> str:
        """Return session_id if it names a single folder; raise ValueError otherwise.

        Rejects empty ids, '.'/'..', path separators and drive markers so that
        no session operation (above all reset_session's rmtree) can address the
        sessions root itself or anything outside it.
        """
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("Session id must be a non-empty string.")
        if session_id in {".", ".."} or any(ch in session_id for ch in ("/", "\\", ":", "\0")):
            raise ValueError(f"Invalid session id: {session_id!r}")
        if session_id != session_id.strip():
            raise ValueError(f"Session id must not start or end with whitespace: {session_id!r}")
        return session_id

    def get_session_dir(self, session_id: str) -> Path:
        """Return absolute path to session directory (a direct child of sessions_root)."""
        self.validate_session_id(session_id)
        sdir = self.sessions_root / session_id
        if sdir.resolve().parent != self.sessions_root.resolve():
            raise ValueError(f"Session id escapes the sessions folder: {session_id!r}")
        return sdir

    def session_exists(self, session_id: str) -> bool:
        """Check if a session folder exists on disk."""
        return self.get_session_dir(session_id).exists()

    def create_session(self, session_id: str, specimen_id: str = "") -> Dict[str, Any]:
        """Create standard folder layout and initialize manifest.json."""
        sdir = self.get_session_dir(session_id)
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "raw").mkdir(exist_ok=True)
        (sdir / "stages").mkdir(exist_ok=True)
        (sdir / "holes").mkdir(exist_ok=True)
        (sdir / "results").mkdir(exist_ok=True)

        manifest = ManifestManager(sdir)
        initial_log = {
            "session_id": session_id,
            "specimen_id": specimen_id,
            "torches": [],
            "camera_settings": {},
            "ball_bearing_used": False,
            "capture_timestamp": datetime.now(timezone.utc).isoformat(),
            "operator_notes": ""
        }
        log_path = sdir / "session_log.json"
        if not log_path.exists():
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(initial_log, f, indent=2)

        return {
            "session_id": session_id,
            "session_dir": str(sdir),
            "manifest": manifest.load()
        }

    def get_manifest_manager(self, session_id: str) -> ManifestManager:
        """Get the ManifestManager for an existing session."""
        sdir = self.get_session_dir(session_id)
        return ManifestManager(sdir)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions on disk and their current manifest status."""
        results = []
        if not self.sessions_root.exists():
            return results

        for child in self.sessions_root.iterdir():
            if child.is_dir():
                manifest_file = child / "manifest.json"
                if manifest_file.exists():
                    try:
                        with open(manifest_file, "r", encoding="utf-8") as f:
                            m = json.load(f)
                        results.append({
                            "session_id": child.name,
                            "status": m.get("status", "unknown"),
                            "created_at": m.get("created_at"),
                            "updated_at": m.get("updated_at"),
                            "holes_total": m.get("holes_total", 0),
                            "holes_processed": m.get("holes_processed", 0)
                        })
                    except Exception:
                        results.append({
                            "session_id": child.name,
                            "status": "corrupt"
                        })
        return results

    def check_incomplete_session(self) -> Optional[Dict[str, Any]]:
        """Scan for a session marked 'in_progress' or 'failed' for crash-resume (Feature 22)."""
        for s in self.list_sessions():
            if s.get("status") in {"in_progress", "failed"}:
                sdir = self.get_session_dir(s["session_id"])
                manifest = ManifestManager(sdir).load()
                return {
                    "session_id": s["session_id"],
                    "status": s["status"],
                    "manifest": manifest,
                    "session_dir": str(sdir)
                }
        return None

    def resume_session(self, session_id: str) -> Dict[str, Any]:
        """Reload session state and return last completed step index."""
        if not self.session_exists(session_id):
            raise FileNotFoundError(f"Session {session_id} not found.")

        manifest = self.get_manifest_manager(session_id).load()
        stages = manifest.get("stages", [])
        completed_steps = [s["step_id"] for s in stages if s.get("status") == "complete"]
        last_step = max(completed_steps) if completed_steps else 0

        return {
            "session_id": session_id,
            "status": manifest.get("status"),
            "last_completed_step": last_step,
            "manifest": manifest
        }

    def reset_session(self, session_id: str) -> bool:
        """Permanently delete session folder on disk (Feature 23).
        
        Requires external confirmation by caller per specification.
        """
        sdir = self.get_session_dir(session_id)  # validates the id
        if sdir.is_dir() and sdir.resolve() != self.sessions_root.resolve():
            shutil.rmtree(sdir)
            return True
        return False

    # Session-relative outputs written by each step; a step's entries (and every
    # later step's) are removed before that step re-runs (Feature 24), so stale
    # files such as surplus boundary patches never survive a reprocess.
    _STEP_OUTPUTS: Dict[int, List[str]] = {
        2: ["stages/calibration.json"],
        3: ["stages/illumination_geometry.json"],
        4: ["stages/brightness_correction.json"],
        5: ["stages/registered", "stages/registration_summary.json"],
        6: ["stages/stamp_frame.json", "stages/stamp_boundary.png"],
        7: ["stages/holes.json", "stages/oversize_candidates.json"],
        9: ["holes"],
        22: ["stages/candidate_regions.json"],
        23: ["results/annotated_overview.png", "results/results_table.csv", "results/report.pdf"],
    }
    # Per-hole outputs (relative to holes/<hole_id>/) by producing step.
    _HOLE_STEP_OUTPUTS: Dict[int, List[str]] = {
        10: ["crops/fused_*.png"],
        11: ["subpixel_outline.json", "contour_overlay.png"],
        12: ["radius_profile.json"],
        13: ["descriptors.json"],
        14: ["boundary_patches"],
        16: ["angular_brightness_table.json"],
        20: ["score.json"],
    }

    def clear_outputs_from_stage(self, session_id: str, step_id: int) -> None:
        """Delete on-disk outputs of step_id and every later step (raw/ and logs are kept)."""
        sdir = self.get_session_dir(session_id)

        def _remove(path: Path) -> None:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink()

        for step, rels in self._STEP_OUTPUTS.items():
            if step >= step_id:
                for rel in rels:
                    _remove(sdir / rel)

        holes_dir = sdir / "holes"
        if holes_dir.is_dir():
            for hole_dir in holes_dir.iterdir():
                if not hole_dir.is_dir():
                    continue
                for step, patterns in self._HOLE_STEP_OUTPUTS.items():
                    if step >= step_id:
                        for pattern in patterns:
                            for path in hole_dir.glob(pattern):
                                _remove(path)

    def save_session_log(self, session_id: str, log_data: Dict[str, Any]) -> bool:
        """Persist structured SessionLog to session_log.json (Feature 2)."""
        sdir = self.get_session_dir(session_id)
        if not sdir.exists():
            raise FileNotFoundError(f"Session {session_id} does not exist.")

        # Ensure elevation is calculated for torches if missing
        torches = log_data.get("torches", [])
        for t in torches:
            if "computed_elevation_deg" not in t:
                h = t.get("height_cm", 0.0)
                d = t.get("distance_cm", 1.0)
                corr = t.get("manual_correction_deg", 0.0)
                t["computed_elevation_deg"] = self.compute_elevation(h, d, corr)

        # Merge rather than replace: keys recorded separately (e.g. Feature 3's
        # shadow_bleed_acknowledged) survive a later save of the form.
        existing = self.load_session_log(session_id) or {}
        merged = {**existing, **log_data}

        log_path = sdir / "session_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        return True

    def load_session_log(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session_log.json if present."""
        sdir = self.get_session_dir(session_id)
        log_path = sdir / "session_log.json"
        if not log_path.exists():
            return None
        with open(log_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def set_shadow_bleed_acknowledged(self, session_id: str, value: bool = True) -> bool:
        """Persist the shadow-bleed acknowledgment boolean on the session record (Feature 3)."""
        sdir = self.get_session_dir(session_id)
        sdir.mkdir(parents=True, exist_ok=True)
        log = self.load_session_log(session_id) or {}
        log["shadow_bleed_acknowledged"] = bool(value)
        log_path = sdir / "session_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        return True

    @staticmethod
    def compute_elevation(height_cm: float, distance_cm: float, manual_correction_deg: float = 0.0) -> float:
        """Compute elevation angle: arctan(height / distance) in degrees + correction (Feature 2)."""
        if distance_cm <= 0:
            return float(manual_correction_deg)
        elev = math.degrees(math.atan(height_cm / distance_cm)) + manual_correction_deg
        return round(elev, 2)

    def store_raw_files(self, session_id: str, tagged: Dict[str, List[str]]) -> Dict[str, int]:
        """Copy tagged capture files into raw/ under canonical names (Feature 1 upload).

        Each tag's files become '<tag><ext>' (one file) or '<tag>_01<ext>',
        '<tag>_02<ext>' ... so Step 1's naming convention always recognises them.
        For every tag being uploaded, older raw/ files of that tag are replaced.
        Sources already inside raw/ are handled (re-tagging works) because all
        copies are staged first. Returns the resulting tag counts in raw/.
        """
        if not self.session_exists(session_id):
            self.create_session(session_id)
        raw_dir = self.get_session_dir(session_id) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        staging = raw_dir / ".staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        try:
            planned: List[str] = []
            # Files already in raw/ under their final name are left in place:
            # RAW captures are ~50 MB each, so re-copying the whole set on every
            # Process click took many seconds for no change.
            kept: set = set()
            for tag, paths in tagged.items():
                if tag not in CAPTURE_SCHEMA:
                    raise ValueError(f"Unknown capture tag: {tag}")
                sources = [Path(p) for p in paths]
                for i, src in enumerate(sources, 1):
                    if not src.is_file():
                        raise FileNotFoundError(f"Capture file not found: {src}")
                    stem = tag if len(sources) == 1 else f"{tag}_{i:02d}"
                    name = stem + src.suffix.lower()
                    if src.resolve() == (raw_dir / name).resolve():
                        kept.add(name)
                        continue
                    shutil.copy2(src, staging / name)
                    planned.append(name)
            uploaded_tags = set(tagged)
            for existing in raw_dir.iterdir():
                if (existing.is_file() and existing.name not in kept
                        and match_filename_to_tag(existing.name) in uploaded_tags):
                    existing.unlink()
            for name in planned:
                shutil.move(str(staging / name), str(raw_dir / name))
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return self.get_tagged_files(session_id)

    def import_raw_folder(self, session_id: str, folder: str) -> Dict[str, int]:
        """Take capture files from a folder by their names (the original workflow).

        Every file whose name follows the capture convention (diffuse.*,
        torch_1.*, backlit_02.*, calibration.*, ball_bearing_torch_3.* ...) is
        copied into raw/; other files are ignored. A session_log.json in the
        folder is merged into the session log when present.
        """
        src_dir = Path(folder)
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")
        tagged: Dict[str, List[str]] = {}
        for entry in sorted(src_dir.iterdir(), key=lambda p: p.name.lower()):
            tag = match_filename_to_tag(entry.name) if entry.is_file() else None
            if tag:
                tagged.setdefault(tag, []).append(str(entry))
        if not tagged:
            raise ValueError("No files in that folder follow the capture naming convention "
                             "(diffuse, torch_1..torch_4, backlit, calibration, ball_bearing_*).")
        counts = self.store_raw_files(session_id, tagged)
        log_file = src_dir / "session_log.json"
        if log_file.is_file():
            with open(log_file, "r", encoding="utf-8") as f:
                log = json.load(f)
            if isinstance(log, dict):
                log["session_id"] = session_id
                self.save_session_log(session_id, log)
        return counts

    def list_raw_files(self, session_id: str) -> List[Dict[str, Any]]:
        """Files currently in raw/ with the tag each one maps to (None if unrecognised)."""
        raw_dir = self.get_session_dir(session_id) / "raw"
        if not raw_dir.is_dir():
            return []
        return [
            {"name": f.name, "path": str(f), "tag": match_filename_to_tag(f.name)}
            for f in sorted(raw_dir.iterdir(), key=lambda p: p.name.lower()) if f.is_file()
        ]

    def get_tagged_files(self, session_id: str) -> Dict[str, int]:
        """Count tagged files in data/sessions/<session_id>/raw/."""
        sdir = self.get_session_dir(session_id)
        raw_dir = sdir / "raw"
        if not raw_dir.exists():
            return {}

        # Same matcher as Step 1, so upload validation and ingestion agree.
        counts: Dict[str, int] = {}
        for f in raw_dir.iterdir():
            if f.is_file():
                tag = match_filename_to_tag(f.name)
                if tag:
                    counts[tag] = counts.get(tag, 0) + 1
        return counts
