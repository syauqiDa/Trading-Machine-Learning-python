from __future__ import annotations

import pandas as pd

from xauusd_ict_ml.config import StrategyConfig, load_config
from xauusd_ict_ml.ict.setup_engine import generate_setups


def test_setup_explanation_records_ablation(sample_bars):
    frame = sample_bars.iloc[:3].copy()
    config = load_config("config/strategy.yaml", StrategyConfig)
    config.setup.enabled_conditions = {name: False for name in config.setup.enabled_conditions}
    structure = pd.DataFrame({"structure_direction": [0]*3, "structure_event": ["none"]*3})
    displacement = pd.DataFrame({"atr": [1.0]*3, "bullish_displacement": [False]*3, "bearish_displacement": [False]*3})
    premium = pd.DataFrame({"zone": ["discount"]*3})
    setups = generate_setups(frame, structure, displacement, pd.DataFrame(), pd.DataFrame(), premium, pd.Series(["new_york"]*3), config)
    assert len(setups) == 6
    assert '"enabled"' in setups.iloc[0]["explanation"]

