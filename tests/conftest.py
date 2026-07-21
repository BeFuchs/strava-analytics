"""Shared fixtures: a minimal FIT encoder that produces real, fitparse-readable files.

Encodes just enough of the FIT binary format (header, definition/data records,
CRC-16) to carry file_id, record and session messages for tests.
"""

from __future__ import annotations

import struct
from datetime import datetime
from pathlib import Path

import pytest

FIT_EPOCH = datetime(1989, 12, 31)
DEFAULT_START = datetime(2024, 5, 1, 8, 0, 0)

SPORT_CYCLING = 2
SPORT_RUNNING = 1

_CRC_TABLE = (
    0x0000,
    0xCC01,
    0xD801,
    0x1400,
    0xF001,
    0x3C00,
    0x2800,
    0xE401,
    0xA001,
    0x6C00,
    0x7800,
    0xB401,
    0x5000,
    0x9C01,
    0x8801,
    0x4400,
)

# name -> (field_def_num, byte size, base type)
_RECORD_FIELDS = {
    "timestamp": (253, 4, 0x86),
    "power": (7, 2, 0x84),
    "heart_rate": (3, 1, 0x02),
    "cadence": (4, 1, 0x02),
    "speed": (6, 2, 0x84),
    "altitude": (2, 2, 0x84),
    "distance": (5, 4, 0x86),
}

_PACK_FMT = {1: "<B", 2: "<H", 4: "<I"}
_INVALID = {0x00: 0xFF, 0x02: 0xFF, 0x84: 0xFFFF, 0x86: 0xFFFFFFFF}


def _crc16(data: bytes, crc: int = 0) -> int:
    for byte in data:
        for nibble in (byte & 0xF, byte >> 4):
            tmp = _CRC_TABLE[crc & 0xF]
            crc = (crc >> 4) & 0x0FFF
            crc = crc ^ tmp ^ _CRC_TABLE[nibble]
    return crc


def _fit_ts(dt: datetime) -> int:
    return int((dt - FIT_EPOCH).total_seconds())


def _definition(local_type: int, global_num: int, fields: list[tuple[int, int, int]]) -> bytes:
    body = struct.pack("<BBHB", 0, 0, global_num, len(fields))
    for num, size, base in fields:
        body += struct.pack("<BBB", num, size, base)
    return bytes([0x40 | local_type]) + body


def _data(local_type: int, fields: list[tuple[int, int, int]], values: list[int | None]) -> bytes:
    out = bytes([local_type])
    for (_, size, base), value in zip(fields, values, strict=True):
        raw = _INVALID[base] if value is None else value
        out += struct.pack(_PACK_FMT[size], raw)
    return out


def _raw_record_value(name: str, value) -> int:
    """Convert a physical value to the raw on-wire integer (FIT scale/offset)."""
    if name == "speed":
        return round(value * 1000)  # m/s, scale 1000
    if name == "altitude":
        return round((value + 500) * 5)  # m, scale 5, offset 500
    if name == "distance":
        return round(value * 100)  # m, scale 100
    return int(value)


def build_fit(
    path: Path,
    records: list[dict],
    *,
    start: datetime = DEFAULT_START,
    sport: int = SPORT_CYCLING,
    include_session: bool = True,
    elapsed_s: float | None = None,
) -> Path:
    """Write a synthetic FIT file.

    ``records``: one dict per record message with physical values (power W,
    heart_rate bpm, cadence rpm, speed m/s, altitude m, distance m). Timestamps
    run at 1 Hz from ``start`` unless a record carries an explicit ``offset_s``.
    """
    start_ts = _fit_ts(start)
    offsets = [rec.get("offset_s", i) for i, rec in enumerate(records)]

    file_id_fields = [(0, 1, 0x00), (1, 2, 0x84), (4, 4, 0x86)]  # type, manufacturer, time_created
    body = _definition(0, 0, file_id_fields)
    body += _data(0, file_id_fields, [4, 1, start_ts])

    rec_names = ["timestamp"] + [
        name
        for name in _RECORD_FIELDS
        if name != "timestamp" and any(name in rec for rec in records)
    ]
    rec_fields = [_RECORD_FIELDS[name] for name in rec_names]
    body += _definition(1, 20, rec_fields)
    for offset, rec in zip(offsets, records, strict=True):
        values: list[int | None] = [start_ts + int(offset)]
        for name in rec_names[1:]:
            values.append(_raw_record_value(name, rec[name]) if name in rec else None)
        body += _data(1, rec_fields, values)

    if include_session:
        if elapsed_s is None:
            elapsed_s = float(max(offsets) - min(offsets)) if offsets else 0.0
        session_fields = [
            (253, 4, 0x86),  # timestamp
            (2, 4, 0x86),  # start_time
            (7, 4, 0x86),  # total_elapsed_time, scale 1000
            (5, 1, 0x00),  # sport
        ]
        last_ts = start_ts + int(max(offsets)) if offsets else start_ts
        body += _definition(2, 18, session_fields)
        body += _data(2, session_fields, [last_ts, start_ts, round(elapsed_s * 1000), sport])

    header12 = struct.pack("<BBHI4s", 14, 0x10, 2132, len(body), b".FIT")
    header = header12 + struct.pack("<H", _crc16(header12))
    content = header + body
    path.write_bytes(content + struct.pack("<H", _crc16(content)))
    return path


@pytest.fixture
def make_fit(tmp_path):
    """Factory fixture: create a synthetic FIT file in tmp_path and return its path."""

    def _make(name: str = "ride.fit", records: list[dict] | None = None, **kwargs) -> Path:
        if records is None:
            records = [{"power": 200, "heart_rate": 140} for _ in range(60)]
        return build_fit(tmp_path / name, records, **kwargs)

    return _make
