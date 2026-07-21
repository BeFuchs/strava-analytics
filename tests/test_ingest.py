from datetime import datetime

import pytest
from conftest import SPORT_RUNNING

from ride_analytics.ingest import IngestError, load_fit, load_rides


def _records(n=60, power=200):
    return [
        {
            "power": power,
            "heart_rate": 140,
            "cadence": 90,
            "speed": 10.0,
            "altitude": 120.0,
            "distance": float(i * 10),
        }
        for i in range(n)
    ]


def test_load_single_cycling_ride(make_fit):
    ride = load_fit(make_fit(records=_records()))

    assert ride is not None
    assert ride.metadata.sport == "cycling"
    assert ride.metadata.source == "ride.fit"
    assert ride.metadata.start_time == datetime(2024, 5, 1, 8, 0, 0)
    assert ride.metadata.duration_s == pytest.approx(59.0)
    assert list(ride.df.columns) == [
        "timestamp",
        "power",
        "heart_rate",
        "cadence",
        "speed",
        "altitude",
        "distance",
    ]
    assert len(ride.df) == 60


def test_units_are_decoded(make_fit):
    ride = load_fit(make_fit(records=_records()))

    assert ride.df["speed"].iloc[0] == pytest.approx(10.0)  # m/s
    assert ride.df["altitude"].iloc[0] == pytest.approx(120.0)  # m
    assert ride.df["distance"].iloc[-1] == pytest.approx(590.0)  # m
    assert ride.df["power"].max() == 200


def test_non_cycling_is_skipped(make_fit, caplog):
    path = make_fit(name="run.fit", sport=SPORT_RUNNING)

    with caplog.at_level("INFO"):
        assert load_fit(path) is None
    assert "run.fit" in caplog.text


def test_timestamps_sorted_and_deduplicated(make_fit):
    records = [
        {"offset_s": 5, "power": 1},
        {"offset_s": 0, "power": 2},
        {"offset_s": 0, "power": 3},
        {"offset_s": 10, "power": 4},
    ]
    ride = load_fit(make_fit(records=records))

    assert len(ride.df) == 3
    assert ride.df["timestamp"].is_monotonic_increasing
    assert ride.df["power"].tolist() == [2, 1, 4]  # first duplicate wins


def test_gaps_are_not_interpolated(make_fit):
    records = [{"offset_s": i, "power": 100} for i in range(10)]
    records += [{"offset_s": 300 + i, "power": 100} for i in range(10)]
    ride = load_fit(make_fit(records=records))

    assert len(ride.df) == 20  # gap stays a gap


def test_missing_power_tolerated(make_fit):
    ride = load_fit(make_fit(records=[{"heart_rate": 150} for _ in range(30)]))

    assert "power" not in ride.df.columns
    assert ride.df["heart_rate"].iloc[0] == 150


def test_load_rides_directory(make_fit, tmp_path):
    make_fit(name="b.fit", records=_records(), start=datetime(2024, 5, 2, 9, 0, 0))
    make_fit(name="a.fit", records=_records(), start=datetime(2024, 5, 1, 9, 0, 0))
    make_fit(name="run.fit", records=_records(), sport=SPORT_RUNNING)

    rides = load_rides(tmp_path)

    assert [r.metadata.source for r in rides] == ["a.fit", "b.fit"]


def test_empty_directory_raises(tmp_path):
    with pytest.raises(IngestError, match="no .fit files"):
        load_rides(tmp_path)


def test_missing_path_raises(tmp_path):
    with pytest.raises(IngestError, match="not found"):
        load_rides(tmp_path / "nope.fit")


def test_corrupt_file_single_raises(tmp_path):
    bad = tmp_path / "bad.fit"
    bad.write_bytes(b"this is not a fit file")

    with pytest.raises(IngestError, match="bad.fit"):
        load_rides(bad)
