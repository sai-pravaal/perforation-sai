"""IPDS Session Manifest Manager.

Single source of truth for session lifecycle, stage completion status,
displayable stage output images, and progress tracking.
Treats manifest.json as an append-mostly log to guarantee crash recovery.
"""

import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Transient sharing violations (Windows refuses to replace a file another
# handle has open, and a reader can catch a write mid-flight) are retried.
_IO_RETRIES = 20
_IO_RETRY_DELAY_S = 0.025

# One re-entrant lock per manifest file, shared by every ManifestManager in this
# process, so concurrent read-modify-write updates (pipeline thread + UI polling)
# cannot interleave and drop each other's changes.
_LOCKS: Dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


# Pipeline step that produces each manifest image "stage" label, used to prune
# image entries on reprocess (Feature 24). Keep in step with add_image_entry callers.
IMAGE_STAGE_STEP: Dict[str, int] = {
    "registration": 5,
    "stamp_boundary": 6,
    "roi_crop": 9,
    "focus_fused": 10,
    "subpixel_contour": 11,
    "boundary_patch": 14,
    "final_output": 23,
}


def _lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        if key not in _LOCKS:
            _LOCKS[key] = threading.RLock()
        return _LOCKS[key]


def _locked(method):
    """Run a read-modify-write manifest method under the per-file lock."""
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    wrapper.__name__ = method.__name__
    wrapper.__doc__ = method.__doc__
    return wrapper


class ManifestReadError(RuntimeError):
    """manifest.json exists but could not be read; never silently replaced."""


class ManifestManager:
    """Manages reading, atomic updating, and diff querying of manifest.json."""

    def __init__(self, session_dir: Path | str) -> None:
        self.session_dir = Path(session_dir)
        self.manifest_path = self.session_dir / "manifest.json"
        self._lock = _lock_for(self.manifest_path)
        self._ensure_manifest()

    def _default_manifest(self) -> Dict[str, Any]:
        """Return the default skeleton for a new session manifest."""
        return {
            "session_id": self.session_dir.name,
            "status": "not_started",
            "stages": [],
            "images": [],
            "holes_total": 0,
            "holes_processed": 0,
            "low_confidence_holes": [],
            "version_metadata": {},
            "mod_token": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

    def _ensure_manifest(self) -> None:
        """Ensure manifest.json exists on disk; create default if absent."""
        with self._lock:
            if not self.manifest_path.exists():
                self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
                self._write_manifest(self._default_manifest())

    def load(self) -> Dict[str, Any]:
        """Read and return the current manifest.json content.

        A missing file yields the default skeleton. A file that exists but
        cannot be read is retried briefly and then raises ManifestReadError:
        returning a default here would let the next write replace the real
        manifest with a blank one.
        """
        if not self.manifest_path.exists():
            return self._default_manifest()
        last_exc: Optional[Exception] = None
        for _ in range(_IO_RETRIES):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                last_exc = exc
                time.sleep(_IO_RETRY_DELAY_S)
        raise ManifestReadError(f"Unable to read {self.manifest_path}: {last_exc}")

    def _write_manifest(self, data: Dict[str, Any]) -> None:
        """Write manifest atomically: temp file + fsync + os.replace.

        os.replace is an atomic rename that overwrites on every platform
        (shutil.move fell back to a non-atomic copy on Windows).
        """
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp_dir = self.manifest_path.parent
        tmp_file = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=tmp_dir, delete=False, suffix=".tmp"
        )
        try:
            json.dump(data, tmp_file, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            tmp_file.close()
            for attempt in range(_IO_RETRIES):
                try:
                    os.replace(tmp_file.name, self.manifest_path)
                    break
                except PermissionError:
                    if attempt == _IO_RETRIES - 1:
                        raise
                    time.sleep(_IO_RETRY_DELAY_S)
        except Exception:
            if os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)
            raise

    @_locked
    def set_status(self, status: str) -> None:
        """Update session-level status (not_started, in_progress, complete, failed)."""
        valid_statuses = {"not_started", "in_progress", "complete", "failed"}
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}. Must be one of {valid_statuses}")
        data = self.load()
        data["status"] = status
        data["mod_token"] += 1
        self._write_manifest(data)

    @_locked
    def add_stage_entry(
        self,
        step_id: int,
        phase: int,
        stage_name: str,
        output_path: str,
        status: str = "complete"
    ) -> None:
        """Record or update a completed or active pipeline step stage."""
        data = self.load()
        stages: List[Dict[str, Any]] = data.get("stages", [])

        # Check if step_id already recorded, update if so
        updated = False
        now_str = datetime.now(timezone.utc).isoformat()
        for s in stages:
            if s.get("step_id") == step_id:
                s["phase"] = phase
                s["stage_name"] = stage_name
                s["output_path"] = output_path
                s["timestamp"] = now_str
                s["status"] = status
                updated = True
                break

        if not updated:
            stages.append({
                "step_id": step_id,
                "phase": phase,
                "stage_name": stage_name,
                "output_path": output_path,
                "timestamp": now_str,
                "status": status
            })

        data["stages"] = stages
        data["mod_token"] += 1
        self._write_manifest(data)

    @_locked
    def add_image_entry(
        self,
        path: str,
        stage: str,
        hole_id: Optional[str] = None,
        edge: Optional[str] = None,
        flagged: Optional[bool] = None
    ) -> int:
        """Append a displayable image entry to the images list.
        
        Returns the new monotonic image token.
        """
        data = self.load()
        images: List[Dict[str, Any]] = data.get("images", [])
        # Tokens must stay monotonic even after invalidate_from_stage prunes
        # entries, so derive them from the ever-increasing mod_token rather
        # than from the list length (which can shrink and repeat a token).
        last = images[-1].get("token", 0) if images else 0
        token = max(int(last), int(data.get("mod_token", 0))) + 1
        # One entry per file: a step re-run (e.g. Steps 1-6 replayed on resume)
        # rewrites the same file, so its old entry is replaced, not duplicated.
        images = [img for img in images if img.get("path") != path]

        entry = {
            "token": token,
            "path": path,
            "stage": stage,
            "hole_id": hole_id,
            "edge": edge,
            "flagged": flagged,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        images.append(entry)
        data["images"] = images
        data["mod_token"] += 1
        self._write_manifest(data)
        return token

    @_locked
    def add_image_entries(self, entries: List[Dict[str, Any]]) -> List[int]:
        """Append many image entries in one write (per-hole outputs would otherwise
        rewrite the manifest once per image). Each entry: path, stage, hole_id, edge."""
        data = self.load()
        images: List[Dict[str, Any]] = data.get("images", [])
        last = images[-1].get("token", 0) if images else 0
        token = max(int(last), int(data.get("mod_token", 0)))
        new_paths = {e["path"] for e in entries}
        images = [img for img in images if img.get("path") not in new_paths]
        tokens: List[int] = []
        now = datetime.now(timezone.utc).isoformat()
        for e in entries:
            token += 1
            images.append({
                "token": token,
                "path": e["path"],
                "stage": e["stage"],
                "hole_id": e.get("hole_id"),
                "edge": e.get("edge"),
                "flagged": e.get("flagged"),
                "timestamp": now,
            })
            tokens.append(token)
        data["images"] = images
        data["mod_token"] = max(int(data.get("mod_token", 0)), token) + 1
        self._write_manifest(data)
        return tokens

    @_locked
    def set_image_flags(self, flagged_by_hole: Dict[str, bool]) -> None:
        """Fill the 'flagged' field of every hole-tagged image once Step 21 has run (Feature 10)."""
        data = self.load()
        for img in data.get("images", []):
            hid = img.get("hole_id")
            if hid in flagged_by_hole:
                img["flagged"] = bool(flagged_by_hole[hid])
        data["mod_token"] += 1
        self._write_manifest(data)

    def get_manifest_diff(self, since_token: int = 0) -> Dict[str, Any]:
        """Return entries added since the specified token for incremental UI refresh."""
        data = self.load()
        all_images: List[Dict[str, Any]] = data.get("images", [])
        new_images = [img for img in all_images if img.get("token", 0) > since_token]
        latest_token = all_images[-1]["token"] if all_images else 0

        return {
            "session_id": data.get("session_id"),
            "status": data.get("status"),
            "latest_token": latest_token,
            "new_images": new_images,
            "holes_total": data.get("holes_total", 0),
            "holes_processed": data.get("holes_processed", 0),
            "stages": data.get("stages", []),
            "mod_token": data.get("mod_token", 0)
        }

    @_locked
    def set_holes_total(self, total: int) -> None:
        """Set the final count of candidate holes (from Step 8)."""
        data = self.load()
        data["holes_total"] = int(total)
        data["mod_token"] += 1
        self._write_manifest(data)

    @_locked
    def increment_holes_processed(self) -> int:
        """Increment running count of processed holes (during Step 18)."""
        data = self.load()
        current = data.get("holes_processed", 0) + 1
        data["holes_processed"] = current
        data["mod_token"] += 1
        self._write_manifest(data)
        return current

    @_locked
    def set_low_confidence_holes(self, hole_ids: List[str]) -> None:
        """Tag holes evaluated under low-confidence fallback (pooled reference)."""
        data = self.load()
        data["low_confidence_holes"] = list(hole_ids)
        data["mod_token"] += 1
        self._write_manifest(data)

    @_locked
    def set_version_metadata(self, metadata: Dict[str, Any]) -> None:
        """Record version stamp and configuration snapshot."""
        data = self.load()
        data["version_metadata"] = metadata
        data["mod_token"] += 1
        self._write_manifest(data)

    @_locked
    def invalidate_from_stage(self, step_id: int) -> None:
        """Invalidate stages and intermediate outputs at or after step_id for Feature 24."""
        data = self.load()
        stages = data.get("stages", [])
        data["stages"] = [s for s in stages if s.get("step_id", 0) < step_id]

        # Drop image entries produced by the invalidated steps so a re-run does
        # not append duplicates (unknown stage names are treated as step 23).
        data["images"] = [
            img for img in data.get("images", [])
            if IMAGE_STAGE_STEP.get(str(img.get("stage")), 23) < step_id
        ]

        # Reset session status if it was previously complete
        if data.get("status") in {"complete", "failed"}:
            data["status"] = "in_progress"

        # If step_id <= 8, reset hole counts
        if step_id <= 8:
            data["holes_total"] = 0
            data["holes_processed"] = 0
            data["low_confidence_holes"] = []
        elif step_id <= 18:
            data["holes_processed"] = 0

        data["mod_token"] += 1
        self._write_manifest(data)
