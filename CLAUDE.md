# nifty_quant

Quantitative research repo for Nifty intraday strategies. Hard rules:

1. All implementation code in this repo is written by DeepSeek (via the `deepseek-worker-agent` role) — the lead/orchestrator does planning, review, and verification only, never hand-writes implementation.
2. Never modify, move, or add anything under `data/` — it is source data, 2.1 GB, already committed to git.
3. float32 at rest, float64 in motion: never accumulate P&L or returns in float32.
4. Bars are left-labelled: the bar labelled T covers the interval [T, T+60s). `ts` is an int64 epoch-seconds UTC value with no timezone attached; convert with `pd.to_datetime(ts, unit="s", utc=True)` then `.tz_convert("Asia/Kolkata")`.
5. Never assume a fixed 375-bar session stride — sessions vary (e.g. Muhurat trading session is 60 bars, disaster-recovery/shortened sessions can be 105 bars). Always index via explicit day offsets, never a fixed row-count-per-day assumption.
6. Never forward-fill bars at rest — NaN means "no bar occurred". Any gap-repair/fill must be opt-in and visible at the call site, never silent.
7. The panel's `present` mask (a bar exists) and `tradable` mask (a bar exists AND is usable for trading, e.g. passes liquidity/halt checks) are distinct concepts and must never be conflated or merged into one.
