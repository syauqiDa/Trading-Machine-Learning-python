from __future__ import annotations

from xauusd_ict_ml.config import DisplacementConfig
from xauusd_ict_ml.ict.displacement import detect_displacement


def test_displacement_combines_range_body_and_close(sample_bars):
    config = DisplacementConfig(atr_period=5, min_range_atr=1.0, min_body_ratio=.5, min_close_location=.6)
    result = detect_displacement(sample_bars, config)
    flagged = result[result["bullish_displacement"]]
    assert (flagged["range_atr"] >= 1).all()
    assert (flagged["body_ratio"] >= .5).all()

