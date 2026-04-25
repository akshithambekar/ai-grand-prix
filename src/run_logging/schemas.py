"""Logging schema placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    qualifier: str
    created_at: float

