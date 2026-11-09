#!/bin/bash -l
#SBATCH -A hpc2n2026-079
#SBATCH -J gpedm_mwcheck
#SBATCH -t 01:30:00
#SBATCH -n 1
#SBATCH -o gpedm_method_tests/logs/checks_missing_week.out
# The series whose newest week is present but listed last (run_stale.py --order, about 3 minutes),
# then the weekly code's choice of locations on past snapshots, the weekly checks on the inputs
# of 12 October, and the scores of the new selection.  Submit from ~/git/Respi.
cd ~/git/Respi
module load foss/2023b matplotlib/3.8.2 R/4.4.1 plotly.py/5.18.0
export OMP_NUM_THREADS=1
PY=/pfs/proj/nobackup/fs/projnb10/forecastmp/Albertas/git/env/test1/bin/python3
$PY gpedm_method_tests/run_stale.py --order
$PY gpedm_method_tests/analyse_stale.py > gpedm_method_tests/results/analysis_stale.txt 2>&1
$PY gpedm_method_tests/check_missing_week.py > gpedm_method_tests/weekly_trial/check_missing_week.log 2>&1
$PY gpedm_method_tests/check_weekly.py > gpedm_method_tests/weekly_trial/check_weekly.log 2>&1
$PY gpedm_method_tests/check_short_series.py > gpedm_method_tests/weekly_trial/check_short_series.log 2>&1
echo done
