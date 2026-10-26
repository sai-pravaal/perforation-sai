"""Full Session Export/Archive (Feature 27).

File: backend/session/archive.py
A thin shutil.make_archive wrapper that packages an already-populated session
directory (raw uploads, stage images, manifest, results table, session log)
into one timestamped zip named from the specimen_id, for reproducibility.
No new data is generated here; the folder is only packaged.

Contains zero Qt/PySide6 imports.
"""

import json
import logging
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


def _safe_name(text: str) -> str:
    """Reduce an arbitrary specimen_id to a filesystem-safe token."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return cleaned or "specimen"


# Already-compressed formats are stored as they are: deflating RAW and PNG files
# saved almost nothing and took ~5x longer (52 s for a 600 MB session).
_STORED_SUFFIXES = {".arw", ".raf", ".cr2", ".cr3", ".nef", ".dng", ".orf", ".rw2", ".png", ".jpg", ".jpeg", ".zip"}


def _write_zip(session_dir: Path, archive: Path) -> None:
    """Zip the session folder with the folder itself as the single top-level entry."""
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for path in sorted(session_dir.rglob("*")):
            arcname = f"{session_dir.name}/{path.relative_to(session_dir).as_posix()}"
            if path.is_dir():
                zf.writestr(arcname + "/", "")
            else:
                kind = zipfile.ZIP_STORED if path.suffix.lower() in _STORED_SUFFIXES else zipfile.ZIP_DEFLATED
                zf.write(path, arcname, compress_type=kind)


class SessionArchiver:
    """Packages a session directory into a single zip archive."""

    def __init__(self, session_manager: Optional[SessionManager] = None) -> None:
        self.session_manager = session_manager or SessionManager()

    def export_session(self, session_id: str, allow_incomplete: bool = False,
                       destination: Optional[str] = None) -> str:
        """Zip the whole session folder and return the archive path.

        destination: where to write the .zip (a file path chosen by the user);
        by default a timestamped file next to the sessions folder. The archive
        keeps the session folder as its single top-level directory, which is
        what import_session expects.
        By default the session manifest must be 'complete'; pass
        allow_incomplete=True to export an in-progress session on explicit
        user request (Feature 27's algorithm step 1).
        """
        session_dir = self.session_manager.get_session_dir(session_id)
        if not session_dir.exists():
            raise FileNotFoundError(f"Session directory not found: {session_id}")

        manifest = self.session_manager.get_manifest_manager(session_id)
        status = manifest.load().get("status", "not_started")
        if status != "complete" and not allow_incomplete:
            raise ValueError(
                f"Session '{session_id}' status is '{status}', not 'complete'. "
                "Pass allow_incomplete=True to export anyway."
            )

        if destination:
            target = Path(destination)
            if target.suffix.lower() == ".zip":
                target = target.with_suffix("")
            target.parent.mkdir(parents=True, exist_ok=True)
            base_name = str(target)
        else:
            log = self.session_manager.load_session_log(session_id) or {}
            specimen_id = _safe_name(str(log.get("specimen_id") or session_id))
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            # Outside the archived tree, which would otherwise recurse.
            base_name = str(session_dir.parent.parent / f"{specimen_id}_{timestamp}")

        archive_path = base_name + ".zip"
        try:
            _write_zip(session_dir, Path(archive_path))
        except Exception:
            Path(archive_path).unlink(missing_ok=True)   # never leave a half-written archive
            raise
        logger.info("Exported session %s to %s", session_id, archive_path)
        return archive_path


class ArchiveError(ValueError):
    """The file is not a usable session archive (message is shown to the user)."""


class SessionExistsError(ValueError):
    """A session with that id is already present; the caller chooses a new id."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"A session named '{session_id}' already exists.")
        self.session_id = session_id


def _top_level_folder(names: List[str]) -> str:
    tops = {n.replace("\\", "/").split("/", 1)[0] for n in names if n.strip("/")}
    if len(tops) != 1:
        raise ArchiveError("This is not a session archive: it must contain exactly one session folder.")
    return tops.pop()


class SessionImporter:
    """Restores a session archive made by SessionArchiver.export_session."""

    def __init__(self, session_manager: Optional[SessionManager] = None) -> None:
        self.session_manager = session_manager or SessionManager()

    def import_session(self, archive_path: str, new_session_id: Optional[str] = None) -> Dict[str, Any]:
        """Unpack the archive as a session and return {session_id, status}.

        The session keeps the folder name stored in the archive unless
        new_session_id is given. Raises SessionExistsError when that id is
        taken, ArchiveError for anything that is not a session archive. The
        archive is checked before anything is written, members that would land
        outside the session folder are rejected, and the session only appears
        once it has been unpacked completely.
        """
        path = Path(archive_path)
        if not path.is_file() or not zipfile.is_zipfile(path):
            raise ArchiveError("That file is not a zip archive.")
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            top = _top_level_folder(names)
            if f"{top}/manifest.json" not in names:
                raise ArchiveError("This zip is not a session archive (no manifest.json inside).")
            session_id = new_session_id or top
            SessionManager.validate_session_id(session_id)
            root = self.session_manager.sessions_root
            if (root / session_id).exists():
                raise SessionExistsError(session_id)

            staging = Path(tempfile.mkdtemp(prefix=".import_", dir=root))
            try:
                for member in zf.infolist():
                    target = (staging / member.filename).resolve()
                    if staging.resolve() not in target.parents and target != staging.resolve():
                        raise ArchiveError("The archive contains files outside the session folder and was not imported.")
                zf.extractall(staging)
                unpacked = staging / top
                if not (unpacked / "manifest.json").is_file():
                    raise ArchiveError("This zip is not a session archive (no manifest.json inside).")
                for sub in ("raw", "stages", "holes", "results"):
                    (unpacked / sub).mkdir(exist_ok=True)
                shutil.move(str(unpacked), str(root / session_id))
            finally:
                shutil.rmtree(staging, ignore_errors=True)

        session_dir = root / session_id
        manifest = self.session_manager.get_manifest_manager(session_id)
        manifest.set_session_id(session_id)
        log = self.session_manager.load_session_log(session_id)
        if isinstance(log, dict) and log.get("session_id") != session_id:
            log["session_id"] = session_id
            with open(session_dir / "session_log.json", "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2)
        logger.info("Imported session %s from %s", session_id, archive_path)
        return {"session_id": session_id, "status": manifest.load().get("status", "not_started")}
