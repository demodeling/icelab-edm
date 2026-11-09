# icelab-edm

Empirical dynamic modelling with Gaussian processes (GP-EDM) for weekly syndromic and
influenza surveillance series, as submitted by IceLab (Integrated Science Lab, Umeå
University) to

- the ECDC [RespiCast syndromic indicators hub](https://github.com/european-modelling-hubs/RespiCast-SyndromicIndicators),
  model `IceLab-EDM`: ARI and ILI incidence, 1 to 4 weeks ahead, 23 quantiles, for every
  reporting country;
- the Swedish national influenza forecast hub, model `dvirnas-edm`: weekly influenza cases
  for Sweden and two regions, point forecasts.

The two names mean the same model. Everything is plain Python on NumPy, pandas, SciPy and
scikit-learn; R is only needed to re-run the historical `vt` method or the validation
against the original R package.

## The model

For one series `x_1 .. x_T` (weekly counts or incidence):

1. `z = log(x + c)`, with `c` a tenth of the series mean and at least 1.
2. The embedding dimension `E` comes from false nearest neighbours at the lag given by the
   first minimum of the mutual information (`efm_params.py`).
3. For each horizon `h = 1..4` a separate Gaussian process with a squared-exponential ARD
   kernel maps the last `E` values `z_s, .., z_(s-E+1)` plus `sin` and `cos` of the time of
   year of the target week to the change `z_(s+h) - z_s`. Hyperparameters maximise the log
   posterior with Rprop from a fixed start, so a fit is deterministic.
4. The forecast is the GP's Gaussian shifted by `z_s` and mapped back through the log: a
   shifted log-normal, cut at zero. RespiCast gets its quantiles, the influenza hub its mean.

Missing weeks before the forecast origin are interpolated linearly. A series whose newest
week is missing is forecast from its last observation with horizons `k+1..k+4`.

The variant was chosen on a replay of 85 RespiCast rounds (October 2024 to October 2026),
see `gpedm_method_tests/`. The mathematics of the GP fit is written out in
`gpedm_py_docs/gpedm_py_math.pdf`.

## Files

| Path | What |
| --- | --- |
| `gpedm_py.py` | NumPy port of `makelags`, `fitGP` and `predict` of the R package [GPEDM](https://github.com/tanyalrogers/GPEDM) (Munch & Rogers), function by function |
| `gpedm_direct.py` | The one-GP-per-horizon forecast on the log scale (the production model) |
| `efm_params.py` | Lag by mutual information, embedding dimension by false nearest neighbours |
| `respi_edm.py` | RespiCast driver: loops over locations, writes the hub's quantile CSV |
| `scoring.py` | Interval score, weighted interval score and point metrics |
| `run_final.ipynb`, `run_final.sbatch` | The weekly RespiCast run (notebook template and its Slurm wrapper) |
| `flu_hub/` | The influenza-hub adaptation: `flu_edm.py` (methods, model versions, pinned code), `live_round.py`, `backtest.py`, `retrospective.py`, the hub metadata and the weekly cron scripts |
| `gpedm_py_docs/` | "How gpedm_py works": LaTeX source, PDF, and the scripts that make its numbers and figures |
| `gpedm_py_validation/` | gpedm_py against the R package: gradient check, synthetic series, every hub series, a full submission file (Python and R outputs side by side) |
| `gpedm_method_tests/` | Replay harness for past RespiCast rounds, the variants tried, and the scored results (`results/`, `README.md`) |
| `greece_analysis/` | Why the Greek series scored badly: notebook, figures and tables |
| `weekly/` | The unattended weekly RespiCast round on a laptop: systemd units, the script that runs the Slurm job and opens the hub pull request, and the prompt of the headless check |

## Running

```bash
pip install -r requirements.txt
python -c "
import pandas as pd, numpy as np, respi_edm as res, gpedm_direct as gd
df = pd.read_csv('gpedm_py_validation/inputs/latest-ARI_incidence.csv')
x, first = res.series_by_date(df[(df.location == 'BE') & df.value.notna()])
E, tau = res.choose_embedding(x)
fc = gd.forecast(x, E, first)
print(E, fc.transform, fc.mean, fc.sd)
"
```

The weekly RespiCast run is `run_final.ipynb`: it reads the hub's `forecasting_weeks.csv`
and `latest-{ARI,ILI}_incidence.csv` from GitHub and appends to
`<origin_date>-IceLab-EDM.csv`. `run_analysis` appends, so delete a partial file before
rerunning. `run_final.sbatch` runs the notebook as a Slurm job on HPC2N's Kebnekaise.

The influenza-hub scripts in `flu_hub/` expect to be run from a project root that holds a
clone of the hub in `hub/` (see their docstrings). They import `gpedm_direct` and
`efm_params` from the parent directory of `flu_hub/`.

`flu_hub/flu_edm.py` pins the md5 of the modules behind each submitted model version;
version 2.0.0 is `gpedm_direct.py` 56646ea0699db888fb476bea9969eb79, `gpedm_py.py`
3640280e8c0fd29dc3f09455117c4f15 and `efm_params.py` d430bdd6b8fda454e4d1eb5592078fc6. These three files are kept byte-identical
to the production copies.

## Dependencies

Python 3.11+, `numpy`, `pandas`, `scipy`, `scikit-learn`; `plotly` for the diagnostic plot
in `efm_params.py`; `matplotlib` for the figures in `gpedm_py_docs/` and
`gpedm_method_tests/`; `rpy2` and the R package GPEDM only for `method="vt"` with
`backend="r"` and for `gpedm_py_validation/`.

## Credits and license

`gpedm_py.py` is a port of GPEDM 0.0.0.9010 by Tanya Rogers and Stephan Munch
(NOAA, US Government work; see `NOTICE`), itself an adaptation of Munch's MATLAB code:

- Munch, S. B., Poynor, V., and Arriaza, J. L. 2017. Circumventing structural uncertainty:
  a Bayesian perspective on nonlinear forecasting for ecology. Ecological Complexity 32:134.
- Johnson, B., and Munch, S. B. 2022. An empirical dynamic modeling framework for missing
  or irregular samples. Ecological Modelling 468:109948.

Input data in `gpedm_py_validation/inputs/` and the truth tables under
`gpedm_method_tests/` are snapshots of the RespiCast hub's public target data (ECDC).

Everything else: Albertas Dvirnas, IceLab, Umeå University, released under CC0 1.0
(`LICENSE`).
