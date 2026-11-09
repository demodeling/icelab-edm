#!/bin/bash -l
#SBATCH -A hpc2n2026-079
#SBATCH -J respi_run_final
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH -C skylake
#SBATCH -o logs/run_final_%j.out
# The weekly run_final notebook as a batch job, for when only the AMD login node
# (kebnekaise-amd) is open: its zen3 module tree cannot load matplotlib/3.8.2.
# Submit from ~/git/Respi:  sbatch run_final.sbatch run_final_YY_MM_DD.ipynb
cd ~/git/Respi
module load foss/2023b matplotlib/3.8.2 R/4.4.1 plotly.py/5.18.0 JupyterLab/4.2.0 jupyterlmod/5.2.1
export OMP_NUM_THREADS=1
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=test1 \
    --ExecutePreprocessor.timeout=1800 "$1"
