#!/bin/bash
#SBATCH --job-name=W4       # create a short name for your job
#SBATCH --partition=quanah
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=2       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --array=0-399:1           # job array with index values
#%module load matlab


module load matlab

# matlab -nodisplay -r "addpath(genpath(\"/home/guiyli/MyTmp/ATPK/\"),genpath(\"/home/guiyli/MyTmp/npy-matlab/npy-matlab/\")); main_function_ATPK_Guiye(128, 4, \"Solar\", $SLURM_ARRAY_TASK_ID, 400, \"_from128\"); exit"

matlab -nodisplay -r "addpath(genpath(\"/home/guiyli/MyTmp/ATPK/\"),genpath(\"/home/guiyli/MyTmp/npy-matlab/npy-matlab/\")); main_function_ATPK_Guiye(128, 4, \"Wind\", $SLURM_ARRAY_TASK_ID, 400, \"_from128\"); exit"

# matlab -nodisplay -r "addpath(genpath(\"/home/guiyli/MyTmp/ATPK/\"),genpath(\"/home/guiyli/MyTmp/npy-matlab/npy-matlab/\")); main_function_ATPK_Guiye(8, 5, \"Solar\", $SLURM_ARRAY_TASK_ID, 400, \"\"); exit"

# matlab -nodisplay -r "addpath(genpath(\"/home/guiyli/MyTmp/ATPK/\"),genpath(\"/home/guiyli/MyTmp/npy-matlab/npy-matlab/\")); main_function_ATPK_Guiye(8, 50, \"Wind\", $SLURM_ARRAY_TASK_ID, 400, \"\"); exit"