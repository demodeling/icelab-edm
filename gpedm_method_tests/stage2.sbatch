#!/bin/bash -l
#SBATCH -A hpc2n2026-079
#SBATCH -J gpedm_stage2
#SBATCH -t 02:00:00
#SBATCH -n 1
#SBATCH --array=0-91
#SBATCH -o gpedm_method_tests/logs/stage2_%a.out
# Second round: the variants in variants.STAGE2, one origin date per task.  Submit from ~/git/Respi
# after the first round, which builds the input cache in cases/.
cd ~/git/Respi
module load foss/2023b matplotlib/3.8.2 R/4.4.1 plotly.py/5.18.0
export OMP_NUM_THREADS=1
/pfs/proj/nobackup/fs/projnb10/forecastmp/Albertas/git/env/test1/bin/python3 gpedm_method_tests/run_replay.py \
    --task $SLURM_ARRAY_TASK_ID --ntasks 92 --variants stage2 --out gpedm_method_tests/fc_stage2 --skip-done
