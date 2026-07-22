"""Pydantic request/response models for the web API."""

from __future__ import annotations

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error: str
    detail: str


class HealthResponse(BaseModel):
    status: str
