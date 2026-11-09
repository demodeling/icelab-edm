#!/bin/bash -l
#SBATCH -A hpc2n2026-079
#SBATCH -J gpedm_gaps_real
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH --array=0-91
#SBATCH -o gpedm_method_tests/logs/gaps_real_%a.out
# Sixth round: missing weeks, the adopted model against its variable-step version (run_gaps.py real).
# Submit from ~/git/Respi.
cd ~/git/Respi
module load foss/2023b matplotlib/3.8.2 R/4.4.1 plotly.py/5.18.0
export OMP_NUM_THREADS=1
/pfs/proj/nobackup/fs/projnb10/forecastmp/Albertas/git/env/test1/bin/python3 gpedm_method_tests/run_gaps.py real \
    --task $SLURM_ARRAY_TASK_ID --ntasks 92
