"""Pydantic request/response models for the web API."""

from __future__ import annotations

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error: str
    detail: str


class HealthResponse(BaseModel):
    status: str


class SkippedFile(BaseModel):
    file: str
    reason: str


class DateRange(BaseModel):
    min: str | None
    max: str | None


class ClusterNameRequest(BaseModel):
    name: str


class UploadResponse(BaseModel):
    session_id: str
    rides_processed: int
    rides_skipped: int
    skip_reasons: list[SkippedFile]
    date_range: DateRange
