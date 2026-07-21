from datetime import date

import pandas as pd
import pytest

from ride_analytics.metrics.pmc import compute_pmc


def rides(*day_tss):
    return pd.DataFrame(
        {
            "date": [date(2024, 5, 1 + d) if d < 30 else date(2024, 6, d - 29) for d, _ in day_tss],
            "tss": [tss for _, tss in day_tss],
        }
    )


def test_first_day_values():
    pmc = compute_pmc(rides((0, 100.0)))

    assert len(pmc) == 1
    assert pmc["ctl"].iloc[0] == pytest.approx(100 / 42)
    assert pmc["atl"].iloc[0] == pytest.approx(100 / 7)
    assert pmc["tsb"].iloc[0] == pytest.approx(0.0)  # no prior load


def test_constant_load_converges_to_tss():
    df = pd.DataFrame(
        {"date": pd.date_range("2022-01-01", periods=1000, freq="D"), "tss": [100.0] * 1000}
    )

    pmc = compute_pmc(df)

    assert pmc["ctl"].iloc[-1] == pytest.approx(100.0, rel=1e-6)
    assert pmc["atl"].iloc[-1] == pytest.approx(100.0, rel=1e-6)
    assert pmc["tsb"].iloc[-1] == pytest.approx(0.0, abs=1e-4)


def test_rest_days_fill_with_zero_tss():
    pmc = compute_pmc(rides((0, 100.0), (4, 50.0)))

    assert len(pmc) == 5
    assert pmc["tss"].tolist() == [100.0, 0.0, 0.0, 0.0, 50.0]


def test_atl_decays_on_rest_days():
    pmc = compute_pmc(rides((0, 100.0), (7, 0.0)))

    expected = (100 / 7) * (1 - 1 / 7) ** 7
    assert pmc["atl"].iloc[7] == pytest.approx(expected)


def test_tsb_is_yesterdays_ctl_minus_atl():
    pmc = compute_pmc(rides((0, 100.0), (3, 80.0)))

    for i in range(1, len(pmc)):
        expected = pmc["ctl"].iloc[i - 1] - pmc["atl"].iloc[i - 1]
        assert pmc["tsb"].iloc[i] == pytest.approx(expected)


def test_same_day_rides_are_summed():
    pmc = compute_pmc(rides((0, 60.0), (0, 40.0)))

    assert len(pmc) == 1
    assert pmc["tss"].iloc[0] == pytest.approx(100.0)
    assert pmc["ctl"].iloc[0] == pytest.approx(100 / 42)


def test_rides_without_tss_are_ignored():
    df = pd.DataFrame({"date": [date(2024, 5, 1), date(2024, 5, 2)], "tss": [100.0, None]})

    pmc = compute_pmc(df)

    assert len(pmc) == 1


def test_empty_input():
    pmc = compute_pmc(pd.DataFrame(columns=["date", "tss"]))

    assert list(pmc.columns) == ["date", "tss", "ctl", "atl", "tsb"]
    assert pmc.empty
