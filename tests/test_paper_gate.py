from __future__ import annotations

import pandas as pd
import pytest

from xauusd_ict_ml.paper import ValidationGate, run_offline_paper_replay


def test_paper_replay_requires_every_gate(sample_bars):
    setups = pd.DataFrame([{"entry_timestamp": sample_bars.at[0, "timestamp"]}])
    gate = ValidationGate(True, True, True, False)
    with pytest.raises(PermissionError):
        run_offline_paper_replay(sample_bars, setups, gate)

