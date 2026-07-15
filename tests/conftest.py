from __future__ import annotations

import pandas as pd
import pytest

from xauusd_ict_ml.data.sample import generate_synthetic_xauusd


@pytest.fixture
def sample_bars() -> pd.DataFrame:
    return generate_synthetic_xauusd(500, seed=7)

