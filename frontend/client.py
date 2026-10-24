"""Frontend API client for the IPDS desktop application.

Error-reporting convention (SESS-06 decision): every backend call returns a
structured result, never an exception, to the UI:
    {"success": True,  "result": <payload>}   or
    {"success": False, "error": "<short, user-readable message>"}
The full technical detail (exception type and traceback) is written to the
application log, not shown to the user. This follows implementation_plan.md
section 3; the frontend specification's alternative (raising ClientError) was
not adopted because an exception escaping a Qt slot can leave a handler
half-finished, while a returned result is always handled in one place
(Client.unwrap) and is easy to explain and test.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Set, Tuple

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from backend import api as backend_api

logger = logging.getLogger(__name__)


def safe_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Wrap a backend call in the structured response; log full detail on failure."""
    name = getattr(func, "__name__", "backend call")
    try:
        result = func(*args, **kwargs)
        if isinstance(result, dict) and "success" in result and "error" in result:
            if result.get("success") is False:
                logger.warning("%s failed: %s", name, result.get("error"))
            return result
        return {"success": True, "result": result}
    except Exception as exc:  # pragma: no cover - defensive UI safety
        logger.exception("Unhandled error in %s", name)
        return {"success": False, "error": f"Unable to complete {name.replace('_', ' ')}: {exc}"}


class _CallRelay(QObject):
    """Lives in the GUI thread; re-delivers a worker result there via a queued signal."""

    done = Signal(object)

    def __init__(self, callback: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        super().__init__()
        self._callback = callback
        self.done.connect(self._deliver)

    def _deliver(self, response: Dict[str, Any]) -> None:
        if self._callback is not None:
            self._callback(response)


class _CallTask(QRunnable):
    def __init__(self, func: Callable[..., Any], args: tuple, kwargs: dict, relay: _CallRelay) -> None:
        super().__init__()
        self._func, self._args, self._kwargs, self.relay = func, args, kwargs, relay

    def run(self) -> None:
        self.relay.done.emit(safe_call(self._func, *self._args, **self._kwargs))


class Client:
    """Thin wrapper used by frontend tabs and widgets to call backend.api.

    Also holds the one piece of client-side state the frontend spec allows:
    the currently active session id, shared by every tab.
    """

    def __init__(self, backend_module: Optional[Any] = None) -> None:
        self.backend = backend_module or backend_api
        self._current_session_id: Optional[str] = None
        self._pending: Set[_CallRelay] = set()

    def call_async(
        self,
        fn_name: str,
        *args: Any,
        on_done: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs: Any,
    ) -> None:
        """Run a long backend call (pipeline run, validation, export) off the GUI thread.

        on_done receives the same structured dict as call(), on the GUI thread.
        """
        func = getattr(self.backend, fn_name, None)
        if func is None:
            if on_done is not None:
                on_done({"success": False, "error": f"Unknown backend action: {fn_name}"})
            return
        relay = _CallRelay(on_done)
        self._pending.add(relay)
        relay.done.connect(lambda _r, rl=relay: self._pending.discard(rl))
        QThreadPool.globalInstance().start(_CallTask(func, args, kwargs, relay))

    def call(self, fn_name: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Execute a backend method through the safe wrapper."""
        func = getattr(self.backend, fn_name, None)
        if func is None:
            return {"success": False, "error": f"Unknown backend action: {fn_name}"}
        return safe_call(func, *args, **kwargs)

    @staticmethod
    def unwrap(response: Dict[str, Any], default_error: str = "The action failed.") -> Tuple[Any, Optional[str]]:
        """Shared response handling for every tab: (payload, None) or (None, error message).

        Treats both the safe_call failure shape and a backend payload that is
        itself a {"success": False, "error": ...} dict as failures, so no view
        reports "loaded" after a failed call.
        """
        if not isinstance(response, dict) or response.get("success") is False:
            return None, (response or {}).get("error") or default_error
        payload = response.get("result", response)
        if isinstance(payload, dict) and payload.get("success") is False:
            return None, payload.get("error") or default_error
        return payload, None

    def set_current_session_id(self, session_id: Optional[str]) -> None:
        self._current_session_id = session_id or None

    def get_current_session_id(self) -> Optional[str]:
        return self._current_session_id
