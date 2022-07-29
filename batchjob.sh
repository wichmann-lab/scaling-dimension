#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=20        # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-12:00            # Runtime in D-HH:MM
#SBATCH --mem=30G                # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --partition=cpu-short
#SBATCH --output=logs/%x_%A_%a.out  # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=logs/%x_%A_%a.err   # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=FAIL            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=david-elias.kuenstle@uni-tuebingen.de   # Email to which notifications will be sent

# print info about current job
scontrol show job $SLURM_JOB_ID
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per no    de with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
echo "Job name: $SLURM_JOB_NAME"

# activate environment and run
conda activate dimensionality
$*  # everything passed after sbatch batchob.sh ...

echo "Finished at $(date)";