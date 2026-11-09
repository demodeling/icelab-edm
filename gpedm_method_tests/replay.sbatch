#!/bin/bash -l
#SBATCH -A hpc2n2026-079
#SBATCH -J gpedm_replay
#SBATCH -t 08:00:00
#SBATCH -n 1
#SBATCH --array=0-367
#SBATCH -o gpedm_method_tests/logs/replay_%a.out
# 92 origin dates x 4 shares of the series of each.  Submit from ~/git/Respi.
cd ~/git/Respi
module load foss/2023b matplotlib/3.8.2 R/4.4.1 plotly.py/5.18.0
export OMP_NUM_THREADS=1
/pfs/proj/nobackup/fs/projnb10/forecastmp/Albertas/git/env/test1/bin/python3 gpedm_method_tests/run_replay.py \
    --task $((SLURM_ARRAY_TASK_ID / 4)) --ntasks 92 --chunk $((SLURM_ARRAY_TASK_ID % 4)) --nchunks 4 --skip-done
