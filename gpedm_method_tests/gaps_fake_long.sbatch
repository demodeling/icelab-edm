#!/bin/bash -l
#SBATCH -A hpc2n2026-079
#SBATCH -J gpedm_gaps_long
#SBATCH -t 01:30:00
#SBATCH -n 1
#SBATCH --array=0-15
#SBATCH -o gpedm_method_tests/logs/gaps_fake_long_%a.out
# run_gaps.py fake for the two origin dates whose snapshots hold 523-week series (2024-10-16, 2024-10-23),
# one series per task (8 chunks per date).  Submit from ~/git/Respi.
cd ~/git/Respi
module load foss/2023b matplotlib/3.8.2 R/4.4.1 plotly.py/5.18.0
export OMP_NUM_THREADS=1
ODS=(2024-10-16 2024-10-23)
/pfs/proj/nobackup/fs/projnb10/forecastmp/Albertas/git/env/test1/bin/python3 gpedm_method_tests/run_gaps.py fake \
    --od ${ODS[$((SLURM_ARRAY_TASK_ID / 8))]} --chunk $((SLURM_ARRAY_TASK_ID % 8)) --nchunks 8
