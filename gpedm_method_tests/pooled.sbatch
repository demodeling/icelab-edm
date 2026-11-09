#!/bin/bash -l
#SBATCH -A hpc2n2026-079
#SBATCH -J gpedm_pooled
#SBATCH -t 06:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --array=0-45
#SBATCH -o gpedm_method_tests/logs/pooled_%a.out
# Third round: the multi-population fit on every second origin date.  Submit from ~/git/Respi.
cd ~/git/Respi
module load foss/2023b matplotlib/3.8.2 R/4.4.1 plotly.py/5.18.0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/pfs/proj/nobackup/fs/projnb10/forecastmp/Albertas/git/env/test1/bin/python3 gpedm_method_tests/run_pooled.py \
    --task $SLURM_ARRAY_TASK_ID --ntasks 46
