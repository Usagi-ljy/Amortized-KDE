"""Privacy-preserving event tracking for the Streamlit application."""

from __future__ import annotations

import logging
import uuid
from typing import Iterable, Optional

import streamlit as st

try:
    from supabase import Client, create_client
except ImportError:  # Keep the app usable if the optional client is unavailable.
    Client = object  # type: ignore[assignment,misc]
    create_client = None


LOGGER = logging.getLogger(__name__)
ALLOWED_EVENTS = {
    "page_view",
    "data_kde_generated",
    "simulation_generated",
}


@st.cache_resource
def _supabase_client() -> Optional[Client]:
    """Create one cached client, or disable tracking when not configured."""

    if create_client is None:
        return None
    try:
        url = str(st.secrets["SUPABASE_URL"]).strip()
        key = str(st.secrets["SUPABASE_ANON_KEY"]).strip()
    except Exception:
        return None
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        LOGGER.exception("Could not initialise the analytics client.")
        return None


def _session_id() -> str:
    """Return an anonymous identifier that lasts only for this app session."""

    key = "_anonymous_analytics_session_id"
    if key not in st.session_state:
        st.session_state[key] = str(uuid.uuid4())
    return str(st.session_state[key])


def track_event(
    event_name: str,
    *,
    workflow: Optional[str] = None,
    distribution: Optional[str] = None,
    sample_size: Optional[int] = None,
    methods: Optional[Iterable[str]] = None,
) -> bool:
    """Insert one anonymous event without interrupting the KDE workflow."""

    if event_name not in ALLOWED_EVENTS:
        raise ValueError(f"Unsupported analytics event: {event_name!r}.")

    client = _supabase_client()
    if client is None:
        return False

    payload = {
        "session_id": _session_id(),
        "event_name": event_name,
        "workflow": workflow,
        "distribution": distribution,
        "sample_size": None if sample_size is None else int(sample_size),
        "methods": None if methods is None else list(methods),
    }
    try:
        client.table("app_events").insert(
            payload,
            returning="minimal",
        ).execute()
        return True
    except Exception:
        LOGGER.exception("Could not record analytics event %s.", event_name)
        return False
