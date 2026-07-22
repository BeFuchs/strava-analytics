"""In-memory session store: parsed rides plus their computed metrics.

Nothing is persisted to disk and there is no database — a server restart
discards all sessions by design (the UI says so). Sessions expire after four
hours without access.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ride_analytics.clustering.climb_clusters import ClimbEffort
from ride_analytics.report.builder import AnalyzedRide

SESSION_TTL = timedelta(hours=4)


@dataclass
class Session:
    session_id: str
    analyzed: list[AnalyzedRide] = field(default_factory=list)
    climb_efforts: list[ClimbEffort] = field(default_factory=list)
    # User-given climb names, keyed by cluster id. Session-scoped like everything
    # else here — never written to disk.
    cluster_names: dict[str, str] = field(default_factory=dict)
    last_access: datetime = field(default_factory=datetime.now)

    def ride_start_times(self) -> set[datetime]:
        return {a.ride.metadata.start_time for a in self.analyzed}


class SessionStore:
    def __init__(self, ttl: timedelta = SESSION_TTL) -> None:
        self._ttl = ttl
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> Session:
        session = Session(session_id=str(uuid.uuid4()))
        with self._lock:
            self._purge_expired()
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        """The session, with its expiry clock reset — ``None`` if unknown/expired."""
        with self._lock:
            self._purge_expired()
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_access = datetime.now()
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _purge_expired(self) -> None:
        cutoff = datetime.now() - self._ttl
        expired = [sid for sid, s in self._sessions.items() if s.last_access < cutoff]
        for sid in expired:
            del self._sessions[sid]
