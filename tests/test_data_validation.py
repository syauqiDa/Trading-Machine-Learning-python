from __future__ import annotations

from xauusd_ict_ml.data.validation import assert_valid_market_data, validate_market_data


def test_valid_sample_has_utc_and_valid_ohlc(sample_bars):
    report = validate_market_data(sample_bars, "15min")
    assert report.invalid_ohlc == 0
    assert report.missing_intervals == 0
    assert report.timezone == "UTC"


def test_duplicate_timestamp_is_rejected(sample_bars):
    sample_bars.loc[1, "timestamp"] = sample_bars.loc[0, "timestamp"]
    report = validate_market_data(sample_bars)
    try:
        assert_valid_market_data(report)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate timestamp was accepted")

