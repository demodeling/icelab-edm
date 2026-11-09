#!/bin/bash -l
#SBATCH -A hpc2n2026-079
#SBATCH -J gpedm_stage4
#SBATCH -t 08:00:00
#SBATCH -n 1
#SBATCH --array=0-367
#SBATCH -o gpedm_method_tests/logs/stage4_%a.out
# Fourth round: the variants in variants.STAGE4 (tuned short + long E mix, neighbouring countries
# as covariates).  92 origin dates x 4 shares of the series of each.  Submit from ~/git/Respi.
cd ~/git/Respi
module load foss/2023b matplotlib/3.8.2 R/4.4.1 plotly.py/5.18.0
export OMP_NUM_THREADS=1
/pfs/proj/nobackup/fs/projnb10/forecastmp/Albertas/git/env/test1/bin/python3 gpedm_method_tests/run_replay.py \
    --task $((SLURM_ARRAY_TASK_ID / 4)) --ntasks 92 --chunk $((SLURM_ARRAY_TASK_ID % 4)) --nchunks 4 \
    --variants stage4 --out gpedm_method_tests/fc_stage4 --skip-done
