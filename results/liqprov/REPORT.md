# Stress-gated residual liquidity-provision fade: backtest report

Generated 2026-09-26T09:33:29+00:00 by scripts/run_liqprov_backtest.py (runtime 99s). Spec: specs/liqprov_fade.md. Pre-registration: prereg.json, config_hash `ee769366e6cea304` (git sha at prereg: c121fae2abe3c01728addaaf72ea0f2b459bc809).

## Caveats (read first)

- Survivorship: universe `nifty50` is the CURRENT Nifty 50 constituent list applied to the whole window, not point-in-time membership. Names that were added to the index during or after the window are traded before they joined it, and names that left the index are absent. Listed mid-window (no data in some window years): ETERNAL (no data through 2020), JIOFIN (no data through 2022), MAXHEALTH (no data through 2019). Missing in 2018: ['ETERNAL', 'JIOFIN', 'MAXHEALTH']. UNIVERSE nifty50 WITH NO AS-OF DATE; 50 names; 3 of 50 names had no data in 2018. Missing names are later listings; this dataset has no delistings.
- Test-window rationale: the hypothesis was generated from exploratory diagnostics on 2023-08 .. 2025-08 (the base VWAP+/-k*ADR fade), so that window is not evidence. The test window [2018-01-01, 2023-08-13] was never used for this family. Bars from 2017-07-17 are loaded only as warm-up for ADR, beta and the rolling gate quantiles; no trade occurs outside usable in-window sessions.
- Holdout untouched: holdout = [2025-08-14, 2026-08-14] from the lock file; the panel ends at 2023-08-11. Holdout read count before/after: 0/0 (record_read never called).
- Slippage: SqrtImpactSlippage() defaults (half_spread 1.5 bps + 10*sqrt(notional / bar traded value)) are ASSUMED, not calibrated (CLAUDE.md rule 8 not satisfied for them). Passive entries pay no slippage and are assumed filled on a one-tick trade-through, with no queue position modelled. Costs: NSEIntradayEquityCosts().
- Tradable mask: tradable_mask on stock columns only, default ADV floor min_adv_inr=5e7 (Rs 5 crore, 20-session trailing mean). That floor is a pre-existing repo CONSTANT, not derived from a measured null (rule 8 not satisfied for it).
- Tick: 0.05 rupees (NSE equity tick for 2018-2023) is used for the passive trade-through test.
- No stop loss: positions exit only on VWAP reversion (next fillable bar's open), 15:20 square-off, or the last bar's close. Long only. Fractional quantities; P&L is summed across symbols with no capital constraint (see max_concurrent).
- Gate warm-up: the rolling gate thresholds need 120 prior usable sessions, so the gate cannot fire before ['2018-01-25'] (first in-window session with finite thresholds); those early sessions are in the denominators as zero-P&L days.
- Gate sparsity: the gate fires on 110 (q=0.80), 41 (q=0.90), 16 (q=0.95) of 1385 usable in-window sessions. Kill criterion c3 (>= 4 positive years out of 6) and the day-clustered t are therefore evaluated on very few trade-days for the stricter gates.
- Unusable sessions (TradingCalendar.is_usable) are untradable: ['2018-11-07', '2019-10-25', '2019-10-27', '2020-11-14', '2021-11-04', '2022-10-24']. The 09:15 bar is excluded from VWAP, TWAP, range, beta and signals.
- Multiple testing: 12 pre-registered trials (3 q x 2 k x 2 modes) at 2 notionals; the best trial is selected in-sample. See the overfitting section.
- Beta/residual decomposition: market part = beta_d * (NIFTY50 open at exit row / open at entry row - 1) in bps; residual = gross_bps - market part. Trades whose index bar is absent at either row are left undecomposed (counted).

## Pre-registered parameters (measured, no P&L)

k grid: minz = session_extreme(z, 'min') per (symbol, session) over 09:16 <= t < 15:20, restricted to usable in-window sessions and finite values; k90 = -quantile(minz, 0.10), k95 = -quantile(minz, 0.05) (numpy 'linear'), rounded to 3 dp. Sample 66301 of 69250 pairs.

| name | quantile of minz | value | k (3 dp) |
|---|---|---|---|
| k90 | 0.10 | -0.548876 | 0.549 |
| k95 | 0.05 | -0.669371 | 0.669 |

Gate: gate = stress_gate(vix_chg, idx_dev_adr, v_day, m_day); v_day = rolling_session_quantile(session_extreme(vix_chg,'max'), usable, q); m_day = rolling_session_quantile(session_extreme(idx_dev_adr,'min'), usable, 1-q); window=250, min_sessions=120; idx_dev_adr = idx_dev / index ADR(N=10). sessions_fired counts in-window usable sessions with the gate True on at least one row (any minute); sessions_fired_trade_window restricts to rows 09:16 <= t < 15:20.

| q | sessions fired | fraction | median v_day | median m_day | first finite |
|---|---|---|---|---|---|
| 0.80 | 110 | 0.0794 | 0.0625 | -0.6176 | 2018-01-25 |
| 0.90 | 41 | 0.0296 | 0.0905 | -0.8409 | 2018-01-25 |
| 0.95 | 16 | 0.0116 | 0.1165 | -1.0541 | 2018-01-25 |

Kill criteria (pre-registered text):

- All criteria are evaluated per trial on net_bps / net_pnl at notional Rs 10L (Rs 1L is a reported sensitivity only), with NSEIntradayEquityCosts() + SqrtImpactSlippage(), over usable sessions in [2018-01-01, 2023-08-13].
- c1_mean_net_positive: mean net_bps over all trades > 0.
- c2_day_t_gt_2: day-clustered t of net_bps (trades collapsed to one mean per session, mean / (std(ddof=1) / sqrt(n_days))) > 2.
- c3_positive_years_ge_4: at least 4 of the 6 calendar years 2018-2023 have summed net_pnl > 0 (a year with no trades is not positive).
- c4_ex_top5_positive: mean net_bps after dropping all trades on the 5 best days by total net_pnl > 0.
- A trial passes only if c1-c4 all hold (any NaN fails).
- Multiple-testing bar: the deflated Sharpe (per-period, sr0 = expected_max_sharpe(effective_n_trials, var of per-period trial Sharpes)) of the best trial at Rs 10L must be >= 0.95. PBO (CSCV, 16 splits) is reported alongside.
- The hypothesis is supported only if at least one trial passes c1-c4 AND the DSR bar is met. Otherwise the family is killed; no re-parameterisation on this window.

## Kill criteria per trial at Rs 1,000,000 (primary)

| trial | mean_net_bps | day_t | positive_years | n_years | mean_net_bps_ex_top5 | c1_mean_net_positive | c2_day_t_gt_2 | c3_positive_years_ge_4 | c4_ex_top5_positive | passes |
|---|---|---|---|---|---|---|---|---|---|---|
| q0.80_k0.549_passive | 19.516 | 2.526 | 4 | 6 | 0.410 | yes | yes | yes | yes | yes |
| q0.80_k0.549_market | 22.986 | 2.043 | 4 | 6 | -3.355 | yes | yes | yes | no | no |
| q0.80_k0.669_passive | 33.499 | 2.474 | 5 | 6 | 4.992 | yes | yes | yes | yes | yes |
| q0.80_k0.669_market | 39.874 | 2.855 | 4 | 6 | 4.272 | yes | yes | yes | yes | yes |
| q0.90_k0.549_passive | 39.504 | 2.915 | 6 | 6 | -1.752 | yes | yes | yes | no | no |
| q0.90_k0.549_market | 48.246 | 2.100 | 5 | 6 | -7.398 | yes | yes | yes | no | no |
| q0.90_k0.669_passive | 50.620 | 2.500 | 6 | 6 | -5.037 | yes | yes | yes | no | no |
| q0.90_k0.669_market | 63.184 | 2.374 | 6 | 6 | -4.644 | yes | yes | yes | no | no |
| q0.95_k0.549_passive | 92.085 | 2.930 | 5 | 6 | -2.529 | yes | yes | yes | no | no |
| q0.95_k0.549_market | 101.945 | 2.107 | 4 | 6 | -8.148 | yes | yes | yes | no | no |
| q0.95_k0.669_passive | 114.303 | 2.066 | 4 | 6 | -18.522 | yes | yes | yes | no | no |
| q0.95_k0.669_market | 127.644 | 2.058 | 4 | 6 | -28.605 | yes | yes | yes | no | no |

Trials passing c1-c4: 3 of 12.

## Kill criteria per trial at Rs 100,000 (sensitivity)

| trial | mean_net_bps | day_t | positive_years | n_years | mean_net_bps_ex_top5 | c1_mean_net_positive | c2_day_t_gt_2 | c3_positive_years_ge_4 | c4_ex_top5_positive | passes |
|---|---|---|---|---|---|---|---|---|---|---|
| q0.80_k0.549_passive | 17.235 | 2.323 | 4 | 6 | -1.996 | yes | yes | yes | no | no |
| q0.80_k0.549_market | 23.142 | 2.056 | 4 | 6 | -3.371 | yes | yes | yes | no | no |
| q0.80_k0.669_passive | 31.120 | 2.312 | 5 | 6 | 2.513 | yes | yes | yes | yes | yes |
| q0.80_k0.669_market | 39.762 | 2.838 | 4 | 6 | 4.022 | yes | yes | yes | yes | yes |
| q0.90_k0.549_passive | 37.059 | 2.722 | 5 | 6 | -4.294 | yes | yes | yes | no | no |
| q0.90_k0.549_market | 48.166 | 2.122 | 5 | 6 | -7.636 | yes | yes | yes | no | no |
| q0.90_k0.669_passive | 48.281 | 2.347 | 6 | 6 | -7.527 | yes | yes | yes | no | no |
| q0.90_k0.669_market | 62.756 | 2.356 | 6 | 6 | -5.152 | yes | yes | yes | no | no |
| q0.95_k0.549_passive | 89.718 | 2.848 | 5 | 6 | -5.104 | yes | yes | yes | no | no |
| q0.95_k0.549_market | 101.633 | 2.105 | 4 | 6 | -8.803 | yes | yes | yes | no | no |
| q0.95_k0.669_passive | 112.065 | 2.005 | 4 | 6 | -20.275 | yes | yes | yes | no | no |
| q0.95_k0.669_market | 127.121 | 2.054 | 4 | 6 | -29.293 | yes | yes | yes | no | no |

Trials passing c1-c4: 2 of 12.

## Summary at Rs 1,000,000

| trial | trades | trades_per_day | win_rate | pct_exit_vwap | pct_exit_square_off | pct_exit_no_bar | mean_gross_bps | mean_net_bps | median_holding_bars | total_gross_pnl | total_net_pnl | total_costs | sharpe_net | sharpe_gross | max_concurrent | worst_trade_net_bps | worst_day_net_pnl | worst_day_date |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q0.80_k0.549_passive | 713 | 0.515 | 0.544 | 0.184 | 0.816 | 0 | 27.921 | 19.516 | 59 | 1990767.833 | 1391479.443 | 599288.390 | 0.599 | 0.840 | 38 | -1087.634 | -401437.787 | 2020-12-21 |
| q0.80_k0.549_market | 575 | 0.415 | 0.536 | 0.174 | 0.826 | 0 | 36.459 | 22.986 | 58 | 2096375.579 | 1321674.275 | 774701.303 | 0.511 | 0.787 | 30 | -1090.165 | -371756.399 | 2020-12-21 |
| q0.80_k0.669_passive | 460 | 0.332 | 0.585 | 0.176 | 0.824 | 0 | 41.767 | 33.499 | 56 | 1921268.383 | 1540966.752 | 380301.631 | 0.718 | 0.875 | 35 | -1087.634 | -289613.523 | 2020-12-21 |
| q0.80_k0.669_market | 361 | 0.261 | 0.571 | 0.155 | 0.845 | 0 | 52.962 | 39.874 | 55 | 1911926.606 | 1439449.375 | 472477.230 | 0.643 | 0.828 | 27 | -1090.165 | -199763.603 | 2020-12-21 |
| q0.90_k0.549_passive | 350 | 0.253 | 0.611 | 0.226 | 0.774 | 0 | 47.678 | 39.504 | 60 | 1668742.145 | 1382642.219 | 286099.927 | 0.634 | 0.746 | 38 | -913.885 | -263678.387 | 2020-08-31 |
| q0.90_k0.549_market | 276 | 0.199 | 0.609 | 0.214 | 0.786 | 0 | 61.385 | 48.246 | 60 | 1694226.009 | 1331600.779 | 362625.230 | 0.510 | 0.630 | 30 | -915.796 | -251740.780 | 2020-12-21 |
| q0.90_k0.669_passive | 236 | 0.170 | 0.644 | 0.208 | 0.792 | 0 | 58.954 | 50.620 | 58.500 | 1391316.887 | 1194622.737 | 196694.150 | 0.569 | 0.648 | 35 | -913.885 | -238612.928 | 2020-08-31 |
| q0.90_k0.669_market | 182 | 0.131 | 0.632 | 0.198 | 0.802 | 0 | 75.820 | 63.184 | 60 | 1379921.032 | 1149939.772 | 229981.260 | 0.519 | 0.605 | 27 | -915.796 | -171445.870 | 2020-08-31 |
| q0.95_k0.549_passive | 187 | 0.135 | 0.722 | 0.203 | 0.797 | 0 | 100.396 | 92.085 | 48 | 1877399.934 | 1721992.979 | 155406.954 | 0.668 | 0.707 | 37 | -374.341 | -209220.556 | 2020-08-31 |
| q0.95_k0.549_market | 145 | 0.105 | 0.676 | 0.200 | 0.800 | 0 | 114.768 | 101.945 | 48 | 1664135.940 | 1478207.917 | 185928.024 | 0.558 | 0.609 | 27 | -376.989 | -158567.131 | 2020-08-31 |
| q0.95_k0.669_passive | 130 | 0.094 | 0.746 | 0.223 | 0.777 | 0 | 122.813 | 114.303 | 48 | 1596567.840 | 1485942.595 | 110625.245 | 0.583 | 0.611 | 33 | -352.183 | -170690.957 | 2020-08-31 |
| q0.95_k0.669_market | 98 | 0.071 | 0.663 | 0.224 | 0.776 | 0 | 140.170 | 127.644 | 48 | 1373669.573 | 1250908.700 | 122760.873 | 0.522 | 0.557 | 25 | -323.142 | -159453.130 | 2020-08-31 |

## Summary at Rs 100,000

| trial | trades | trades_per_day | win_rate | pct_exit_vwap | pct_exit_square_off | pct_exit_no_bar | mean_gross_bps | mean_net_bps | median_holding_bars | total_gross_pnl | total_net_pnl | total_costs | sharpe_net | sharpe_gross | max_concurrent | worst_trade_net_bps | worst_day_net_pnl | worst_day_date |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q0.80_k0.549_passive | 713 | 0.515 | 0.530 | 0.184 | 0.816 | 0 | 27.921 | 17.235 | 59 | 199076.783 | 122883.196 | 76193.588 | 0.529 | 0.840 | 38 | -1091.484 | -41115.426 | 2020-12-21 |
| q0.80_k0.549_market | 575 | 0.415 | 0.534 | 0.174 | 0.826 | 0 | 36.459 | 23.142 | 58 | 209637.558 | 133063.894 | 76573.664 | 0.513 | 0.787 | 30 | -1093.310 | -37433.477 | 2020-12-21 |
| q0.80_k0.669_passive | 460 | 0.332 | 0.574 | 0.176 | 0.824 | 0 | 41.767 | 31.120 | 56 | 192126.838 | 143151.007 | 48975.831 | 0.669 | 0.875 | 35 | -1091.484 | -29848.272 | 2020-12-21 |
| q0.80_k0.669_market | 361 | 0.261 | 0.571 | 0.155 | 0.845 | 0 | 52.962 | 39.762 | 55 | 191192.661 | 143539.477 | 47653.184 | 0.641 | 0.828 | 27 | -1093.310 | -20148.579 | 2020-12-21 |
| q0.90_k0.549_passive | 350 | 0.253 | 0.597 | 0.226 | 0.774 | 0 | 47.678 | 37.059 | 60 | 166874.215 | 129706.900 | 37167.314 | 0.598 | 0.746 | 38 | -917.723 | -27331.184 | 2020-08-31 |
| q0.90_k0.549_market | 276 | 0.199 | 0.601 | 0.214 | 0.786 | 0 | 61.385 | 48.166 | 60 | 169422.601 | 132938.597 | 36484.004 | 0.508 | 0.630 | 30 | -919.353 | -25409.280 | 2020-12-21 |
| q0.90_k0.669_passive | 236 | 0.170 | 0.636 | 0.208 | 0.792 | 0 | 58.954 | 48.281 | 58.500 | 139131.689 | 113942.776 | 25188.912 | 0.545 | 0.648 | 35 | -917.723 | -24549.411 | 2020-08-31 |
| q0.90_k0.669_market | 182 | 0.131 | 0.621 | 0.198 | 0.802 | 0 | 75.820 | 62.756 | 60 | 137992.103 | 114215.026 | 23777.077 | 0.516 | 0.605 | 27 | -919.353 | -17441.880 | 2020-08-31 |
| q0.95_k0.549_passive | 187 | 0.135 | 0.706 | 0.203 | 0.797 | 0 | 100.396 | 89.718 | 48 | 187739.993 | 167771.986 | 19968.008 | 0.655 | 0.707 | 37 | -377.873 | -21883.819 | 2020-08-31 |
| q0.95_k0.549_market | 145 | 0.105 | 0.683 | 0.200 | 0.800 | 0 | 114.768 | 101.633 | 48 | 166413.594 | 147368.015 | 19045.579 | 0.556 | 0.609 | 27 | -379.736 | -16300.680 | 2020-08-31 |
| q0.95_k0.669_passive | 130 | 0.094 | 0.723 | 0.223 | 0.777 | 0 | 122.813 | 112.065 | 48 | 159656.784 | 145684.719 | 13972.065 | 0.574 | 0.611 | 33 | -355.713 | -17756.180 | 2020-08-31 |
| q0.95_k0.669_market | 98 | 0.071 | 0.663 | 0.224 | 0.776 | 0 | 140.170 | 127.121 | 48 | 137366.957 | 124579.063 | 12787.895 | 0.520 | 0.557 | 25 | -325.988 | -16245.567 | 2020-08-31 |

## March-2020 net P&L per trial (Rs)

| trial | Rs 1,000,000 | Rs 100,000 |
|---|---|---|
| q0.80_k0.549_passive | 414494.342 | 40545.537 |
| q0.80_k0.549_market | 256277.412 | 25986.453 |
| q0.80_k0.669_passive | 405881.375 | 39998.127 |
| q0.80_k0.669_market | 269287.362 | 27147.702 |
| q0.90_k0.549_passive | 428112.953 | 42079.011 |
| q0.90_k0.549_market | 274702.876 | 27614.069 |
| q0.90_k0.669_passive | 384272.528 | 37973.270 |
| q0.90_k0.669_market | 244112.522 | 24435.172 |
| q0.95_k0.549_passive | 520709.680 | 51487.599 |
| q0.95_k0.549_market | 313150.657 | 31533.723 |
| q0.95_k0.669_passive | 343244.375 | 34061.498 |
| q0.95_k0.669_market | 205938.501 | 20627.096 |

## Per-year net P&L at Rs 1,000,000 (Rs)

| trial | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|---|
| q0.80_k0.549_passive | 865215.446 | 112873.254 | -209096.332 | 306845.943 | -107802.035 | 423443.167 |
| q0.80_k0.549_market | 1093757.557 | 107965.109 | -241106.016 | 193048.006 | -115838.029 | 283847.648 |
| q0.80_k0.669_passive | 954374.119 | 134939.485 | -35609.333 | 200440.733 | 25415.923 | 261405.826 |
| q0.80_k0.669_market | 1006193.919 | 112966.671 | -15459.006 | 154166.812 | -727.616 | 182308.595 |
| q0.90_k0.549_passive | 756143.087 | 54001.838 | 33396.158 | 198676.712 | 83393.902 | 257030.521 |
| q0.90_k0.549_market | 1025362.045 | 54422.128 | -36573.351 | 112343.792 | 10886.348 | 165159.816 |
| q0.90_k0.669_passive | 783918.855 | 39237.883 | 118448.607 | 113731.848 | 20279.782 | 119005.763 |
| q0.90_k0.669_market | 887786.597 | 47118.684 | 10285.614 | 84205.499 | 19925.207 | 100618.172 |
| q0.95_k0.549_passive | 990234.064 | 68368.650 | 587913.902 | 45589.568 | 45553.389 | -15666.594 |
| q0.95_k0.549_market | 1091896.725 | 67101.450 | 330003.848 | 7223.065 | -7585.884 | -10431.287 |
| q0.95_k0.669_passive | 1029582.419 | 43694.139 | 418363.183 | 18586.919 | -14471.227 | -9812.839 |
| q0.95_k0.669_market | 983874.617 | 42850.189 | 243633.952 | 9572.634 | -19493.755 | -9528.937 |

## Per-year net P&L at Rs 100,000 (Rs)

| trial | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|---|
| q0.80_k0.549_passive | 84972.934 | 10046.567 | -25645.602 | 26969.322 | -13857.043 | 40397.018 |
| q0.80_k0.549_market | 111837.528 | 10988.320 | -23889.735 | 18328.704 | -12176.492 | 27975.568 |
| q0.80_k0.669_passive | 93938.473 | 12905.945 | -7004.042 | 18004.908 | 405.529 | 24900.194 |
| q0.80_k0.669_market | 101821.582 | 11328.267 | -1698.304 | 14870.376 | -631.636 | 17849.192 |
| q0.90_k0.549_passive | 74616.761 | 5193.428 | -413.886 | 18738.320 | 6903.460 | 24668.818 |
| q0.90_k0.549_market | 102665.365 | 5472.697 | -4124.473 | 11197.960 | 1106.904 | 16620.144 |
| q0.90_k0.669_passive | 77589.759 | 3831.333 | 9038.206 | 10903.596 | 1237.248 | 11342.634 |
| q0.90_k0.669_market | 88574.783 | 4719.380 | 709.525 | 8417.397 | 1858.512 | 9935.429 |
| q0.95_k0.549_passive | 98372.644 | 6781.594 | 55981.870 | 4353.135 | 3937.807 | -1655.063 |
| q0.95_k0.549_market | 109219.730 | 6710.751 | 32579.819 | 763.024 | -887.869 | -1017.441 |
| q0.95_k0.669_passive | 102420.179 | 4327.070 | 39866.327 | 1748.465 | -1667.536 | -1009.786 |
| q0.95_k0.669_market | 98364.654 | 4279.869 | 23925.607 | 997.175 | -2049.990 | -938.252 |

## Beta / residual decomposition at Rs 1,000,000 (mean bps per trade)

| trial | mean_gross_bps | mean_market_bps | mean_residual_bps | n_decomposed | n_undecomposed |
|---|---|---|---|---|---|
| q0.80_k0.549_passive | 27.921 | -3.143 | 31.065 | 713 | 0 |
| q0.80_k0.549_market | 36.459 | -0.633 | 37.092 | 575 | 0 |
| q0.80_k0.669_passive | 41.767 | -3.212 | 44.979 | 460 | 0 |
| q0.80_k0.669_market | 52.962 | 1.691 | 51.271 | 361 | 0 |
| q0.90_k0.549_passive | 47.678 | -2.975 | 50.653 | 350 | 0 |
| q0.90_k0.549_market | 61.385 | 1.464 | 59.921 | 276 | 0 |
| q0.90_k0.669_passive | 58.954 | -6.468 | 65.422 | 236 | 0 |
| q0.90_k0.669_market | 75.820 | -0.053 | 75.872 | 182 | 0 |
| q0.95_k0.549_passive | 100.396 | 10.600 | 89.796 | 187 | 0 |
| q0.95_k0.549_market | 114.768 | 16.889 | 97.879 | 145 | 0 |
| q0.95_k0.669_passive | 122.813 | 12.531 | 110.282 | 130 | 0 |
| q0.95_k0.669_market | 140.170 | 13.492 | 126.679 | 98 | 0 |

## Overfitting statistics (Rs 1,000,000)

Trial matrix: daily net P&L / notional, 1385 usable in-window sessions x 12 trials (zero days included).

| statistic | value |
|---|---|
| effective_n_trials | 1.239 |
| var of per-period trial Sharpes | 1.8386e-05 |
| best trial (annualised sharpe_net) | q0.80_k0.669_passive |
| best annualised Sharpe | 0.718 |
| best per-period Sharpe | 0.045 |
| sr0 (per-period) | 0 |
| deflated Sharpe (probability) | 0.989 |
| DSR bar (0.95) met | yes |
| PBO (CSCV, 16 splits) | 0.733 |

sr0 note: effective_n_trials=1.2390 < 2 (or NaN), or fewer than 2 varying trials: no multiple-testing penalty applied, sr0=0.0 (expected_max_sharpe requires n >= 2).

Verdict against the pre-registration: 3 of 12 trials pass c1-c4 at Rs 1,000,000; DSR of the best trial = 0.9889 vs bar 0.95. Hypothesis SUPPORTED.

## 10 worst trades at Rs 1,000,000 (distinct trades across trials)

| symbol | entry (IST) | exit (IST) | entry_price | exit_price | exit_reason | gross_bps | net_bps | market_bps | first trial | trials containing it |
|---|---|---|---|---|---|---|---|---|---|---|
| ADANIENT | 2023-01-27 12:18 | 2023-01-27 15:20 | 2986.30 | 2663.18 | square_off | -1082.0 | -1090.2 | 2.4 | q0.80_k0.549_market | 4 |
| ADANIENT | 2023-01-27 12:33 | 2023-01-27 15:20 | 2929.20 | 2663.18 | square_off | -908.2 | -915.8 | 33.4 | q0.90_k0.549_market | 4 |
| ADANIENT | 2023-02-01 14:51 | 2023-02-01 15:20 | 2162.39 | 1973.14 | square_off | -875.2 | -884.8 | 330.0 | q0.80_k0.549_market | 4 |
| BEL | 2020-12-21 13:39 | 2020-12-21 15:20 | 39.41 | 37.43 | square_off | -502.8 | -510.4 | -196.6 | q0.80_k0.549_passive | 1 |
| BEL | 2020-12-21 13:40 | 2020-12-21 15:20 | 39.37 | 37.43 | square_off | -492.8 | -505.3 | -196.4 | q0.80_k0.549_market | 1 |
| SHRIRAMFIN | 2020-08-31 11:13 | 2020-08-31 15:20 | 147.68 | 140.38 | square_off | -494.3 | -501.2 | -307.1 | q0.80_k0.549_passive | 1 |
| SHRIRAMFIN | 2020-08-31 11:14 | 2020-08-31 15:20 | 147.46 | 140.38 | square_off | -480.1 | -490.3 | -299.1 | q0.80_k0.549_market | 1 |
| BAJAJFINSV | 2020-08-31 11:18 | 2020-08-31 15:20 | 647.87 | 618.58 | square_off | -452.2 | -458.8 | -267.1 | q0.80_k0.549_passive | 1 |
| BEL | 2020-12-21 13:55 | 2020-12-21 15:20 | 39.16 | 37.43 | square_off | -440.7 | -448.2 | -182.0 | q0.80_k0.669_passive | 1 |
| BEL | 2020-12-21 13:56 | 2020-12-21 15:20 | 39.12 | 37.43 | square_off | -432.0 | -442.9 | -183.6 | q0.80_k0.669_market | 1 |
