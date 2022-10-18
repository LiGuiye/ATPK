#!/bin/bash
#SBATCH --job-name=ATPK-4S       # create a short name for your job
#SBATCH --partition=quanah
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=36       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --array=0-19:1           # job array with index values
#%module load matlab


matlab -nodisplay -r "addpath(genpath(\"/home/guiyli/Documents/ATPK/\"),genpath(\"/home/guiyli/Documents/npy-matlab/npy-matlab/\")); main_function_ATPK_Guiye(4, Solar, $SLURM_ARRAY_TASK_ID); exit"


