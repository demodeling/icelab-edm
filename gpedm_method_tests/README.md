# gpedm_method_tests

Replay of past RespiCast rounds with variants of the GP-EDM forecast, all fitted with `gpedm_py.py`.
Each variant forecasts from the hub snapshot of the Friday before the origin date and is scored
against the values published later and against the other hub models.

Run from `~/git/Respi` on Kebnekaise, with the modules and the `test1` interpreter of the weekly run.

| Step | Command | Output |
| --- | --- | --- |
| Hub data | `git clone --depth 1 --filter=blob:none --sparse https://github.com/european-modelling-hubs/RespiCast-SyndromicIndicators.git hub`, then `git -C hub sparse-checkout set target-data supporting-files model-output/IceLab-EDM model-output/respicast-hubEnsemble`; the score file is `model-evaluation/latest-forecast_scores.csv` of the same repository | `hub/`, `latest-forecast_scores.csv` |
| Check the replay against the submitted forecasts | `run_replay.py --od <dates> --variants prod_submitted --out gpedm_method_tests/fc_validate`, then `validate.py` | stdout |
| Which truth the hub scores against | `check_truth.py` | `truth_consistent.csv` |
| Round 1, all variants in `variants.VARIANTS` | `sbatch gpedm_method_tests/replay.sbatch` | `fc/` |
| Round 2, `variants.STAGE2` | `sbatch gpedm_method_tests/stage2.sbatch` | `fc_stage2/` |
| Round 3, `variants.STAGE3` | `sbatch gpedm_method_tests/stage3.sbatch` | `fc_stage3/` |
| One GP for all countries, every second origin date | `sbatch gpedm_method_tests/pooled.sbatch` | `fc_pooled/` |
| Round 4, `variants.STAGE4` and `STAGE4B`: short + long E members, neighbouring countries as covariates (`neighbours.py`) | `sbatch gpedm_method_tests/stage4.sbatch`, `stage4b.sbatch` | `fc_stage4/`, `fc_stage4b/` |
| Round 5, series whose latest week is missing, horizons shifted (`run_stale.py`) | `sbatch gpedm_method_tests/stale.sbatch` | `fc_stale/` |
| Series whose newest week is present but whose rows are listed oldest first (UK nations, Swiss ILI), which the first-row test skipped up to round 2026-10-14 | `run_stale.py --order` (inside `checks_missing_week.sbatch`) | `fc_order/` |
| Round 6, missing weeks: the adopted model (lag weeks interpolated) against its variable-step version (`vt_direct.py`: the last E observed weeks and the weeks between them), on the weekly run's series with a gap (`real`) and on gap-free series with weeks removed at random (`fake`) | `sbatch gpedm_method_tests/gaps_real.sbatch`, `gaps_fake.sbatch` (`run_gaps.py`); the two origin dates whose snapshots hold 523-week series (2024-10-16, 2024-10-23) take about 2 hours each in `fake` mode, so they run one series per task with `gaps_fake_long.sbatch` | `fc_gaps_real/`, `fc_gaps_fake/` |
| Completeness | `check_run.py` | stdout |
| Scores | `analyse.py`, `analyse_detail.py`, `analyse_offset.py`, `analyse_pooled.py`, `analyse_mix.py`, `analyse_neighbours.py`, `analyse_stale.py`, `analyse_gaps.py`, `make_figure.py` | `results/` |
| The weekly run | `check_weekly.py`: `respi_edm.run_analysis` with its defaults against the replayed model and the submitted file; `check_short_series.py`: locations with very little data are skipped, not fatal; `check_missing_week.py`: on six past snapshots, the locations the weekly run picks (latest date at most `respi_edm.MAX_WEEKS_MISSING` = 1 week old) and their forecasts against the replay. All three run in `sbatch gpedm_method_tests/checks_missing_week.sbatch` | `weekly_trial/` |

`variants.py` defines the variants, `replay_data.py` builds the inputs, `replay_scoring.py` holds the quantile and WIS functions.
The one-GP-per-horizon code itself is the production module `../gpedm_direct.py`, which `respi_edm.run_analysis(method="direct")` submits from round 2026-10-21; the variant adopted is `direct_log_diff_season_off10`.
`results/scored.csv.gz` (every forecast with its scores, 50 MB) is kept on Kebnekaise only.
