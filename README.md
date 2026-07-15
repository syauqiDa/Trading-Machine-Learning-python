# XAUUSD ICT-style ML research project

This repository converts selected ideas inspired by the ICT 2022 mentorship into deterministic, configurable research rules. It is for education, backtesting, reproducible experiments, and gated paper replay only. It does **not** connect to a broker, place live orders, or claim that any strategy is profitable.

## Design guarantees

- Every swing stores occurrence and later confirmation times; trading sees only confirmations.
- A fair value gap becomes available only when its third candle has closed.
- Prior-day and prior-week levels use completed periods only.
- Higher-timeframe bars are right-closed and joined backward after completion.
- Feature windows end at the setup decision timestamp. Labels are generated separately from future bars.
- Preprocessing lives inside scikit-learn pipelines fitted on training rows only.
- All splits are chronological and support label-window purging plus embargo.
- A same-bar stop/target collision defaults to stop first and is counted as ambiguous.
- If `lower_timeframe_path` is configured, ordered sub-bars resolve a parent-candle collision before the conservative fallback is used.
- Longs enter at ask and exit at bid; shorts enter at bid and exit at ask. Synthetic spread is used only when quotes are absent.
- Missing OHLC intervals are never forward-filled unless explicitly enabled.

## Installation

Python 3.11 or newer is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,boosting]'
```

XGBoost is optional. If it is not installed, the training workflow logs a warning and continues with dummy, logistic, random-forest, and gradient-boosting baselines.
On macOS, the XGBoost wheel also requires the system OpenMP runtime (`brew install libomp`). If that native runtime is missing, the optional model is skipped and the required scikit-learn baselines continue.

## Complete synthetic pipeline

```bash
python scripts/prepare_data.py --config config/data.yaml --generate-sample
python scripts/detect_setups.py --config config/strategy.yaml
python scripts/run_backtest.py --config config/backtest.yaml
python scripts/build_dataset.py --config config/model.yaml
python scripts/train_model.py --config config/model.yaml
python scripts/run_walk_forward.py --config config/model.yaml
python scripts/generate_report.py
pytest -q
```

Outputs are written to `data/processed/`, `models/`, `reports/`, and `logs/`. Each training run receives a unique UTC timestamped directory and a provenance manifest.

The bundled synthetic model configuration maps time-barrier/neutral outcomes to class `0` so the smoke-test dataset remains runnable. For research where “neither stop nor target” is substantively distinct, change `neutral_policy` to `drop` or `separate` before freezing the experiment design.

## Supplying real XAUUSD data

Place private CSV or Parquet files under `data/raw/` (this directory is git-ignored), then update `input_path`, `format`, column mappings, `source_timezone`, and `expected_frequency` in `config/data.yaml`. Required fields are timestamp and OHLC. Volume, bid, ask, and spread are optional; their absence triggers an execution-reliability warning.

Never specify a fixed UTC offset for London or New York sessions. Use an IANA name such as `Europe/London` or `America/New_York`; pandas then handles daylight-saving transitions. The configured source timezone applies only to naive input timestamps. Already timezone-aware timestamps are converted directly to UTC.

If a lower-timeframe file is available for execution-path resolution, prepare it with the same UTC schema and set `lower_timeframe_path` in `config/backtest.yaml`. Leave it `null` to retain the documented stop-first candle fallback.

## Deterministic ICT-inspired assumptions

These are research definitions, not assertions that a discretionary concept has one canonical meaning.

- **Swing:** strict high/low beyond `left_bars` and `right_bars`; confirmed after the right-side bars close.
- **Structure:** a close beyond the latest confirmed swing plus an ATR buffer. Same-state breaks are BOS; opposite-state breaks are tagged MSS and CHOCH. The latest opposite confirmed pivot is the protected level.
- **Liquidity:** completed prior day/week extremes and confirmed swing levels. Equal pivots are recognized when their confirmed prices fall within tick, percentage, or ATR tolerance.
- **Sweep:** an excursion beyond a previously known level followed by a close back across it within the configured reclaim bars.
- **Displacement:** configurable conjunction of range/ATR, body ratio, close location, optional volume expansion, optional consecutive candles, and optional confirmed structure break.
- **FVG:** standard three-candle gap, stamped at candle three close, with touches/fills/invalidation tracked only afterward.
- **Premium/discount:** trailing completed dealing-range high/low and midpoint. No future range endpoint is used.
- **Order block:** the last opposing candle before a displacement that also breaks confirmed structure. Full-candle or body boundaries and maximum age are configurable.
- **Setup:** mirrored bullish/bearish conditions are ablatable in `strategy.yaml`. The conservative stop sits beyond the swept extreme plus an ATR buffer; the default target is fixed risk/reward.

## Safe configuration changes

Edit one YAML group at a time and rerun `pytest -q` plus the complete pipeline. Unknown keys are rejected by Pydantic. Keep a new final test period untouched whenever thresholds, features, or rule parameters change. Never choose a threshold using final-test results. After changing a temporal definition, specifically inspect the swing, FVG, resampling, feature-leakage, and walk-forward tests.

Recommended sensitivity workflow:

1. Choose a small, predeclared parameter grid using training/validation only.
2. Run cost, spread, and slippage stress cases.
3. Run ablations, label shuffling, random-count controls, and trade-order reshuffling.
4. Examine stability by year, session, direction, and volatility regime.
5. Freeze everything before evaluating the untouched final period.

## Reports and limitations

The pipeline saves complete trades, equity, drawdown, grouped performance, classification/calibration metrics, probability buckets, walk-forward folds, and HTML/PNG/CSV outputs. Treat small samples, narrow parameter optima, single-year or single-regime concentration, poor calibration, large train/test gaps, and cost-sensitive results as warnings rather than discoveries.

Remaining limitations include candle-level path ambiguity when lower-timeframe data is absent, synthetic sample data that cannot represent broker microstructure, simplified financing/exposure accounting, no latency or queue-position simulation, and no economic-news feed. Optional XGBoost availability depends on the local environment. Order blocks and market-structure terminology remain approximations whose exact assumptions are deliberately visible in configuration.

## Paper trading gate

The code contains no live-broker integration or API keys. Any later paper-trading adapter should consume only closed bars, use the same execution model, and remain disabled until the full test suite, data-quality checks, synthetic pipeline, walk-forward validation, and a user-approved validation checklist pass. Live trading is intentionally out of scope.

An offline paper-replay command is included. It only writes candidate alerts and requires a user-created JSON gate with `tests_passed`, `data_quality_passed`, `walk_forward_completed`, and `user_approved` all set to `true`:

```bash
python scripts/paper_replay.py --gate validation_gate.json
```

The replay status is always `candidate_only_no_order_sent`; there is no network or broker code behind this command.
# Trading-Machine-Learning-python
