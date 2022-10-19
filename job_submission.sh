#!/bin/bash
#SBATCH --job-name=ATPK-8W       # create a short name for your job
#SBATCH --partition=quanah
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=20       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --array=0-1:1           # job array with index values
#%module load matlab


module load matlab
matlab -nodisplay -r "addpath(genpath(\"/home/guiyli/MyTmp/ATPK/\"),genpath(\"/home/guiyli/MyTmp/npy-matlab/npy-matlab/\")); main_function_ATPK_Guiye(8, \"Wind\", $SLURM_ARRAY_TASK_ID, 2); exit"