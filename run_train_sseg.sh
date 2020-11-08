#!/bin/sh

## Give your job a name to distinguish it from other jobs you run.
#SBATCH --job-name=uncertainty_classification

## General partitions: all-HiPri, bigmem-HiPri   --   (12 hour limit)
##                     all-LoPri, bigmem-LoPri, gpuq  (5 days limit)
## Restricted: CDS_q, CS_q, STATS_q, HH_q, GA_q, ES_q, COS_q  (10 day limit)

#SBATCH --partition=gpuq        # Default is all-HiPri
#SBATCH --constraint=gpu-k80

## Separate output and error messages into 2 files.
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID
#SBATCH --output=/scratch/yli44/logs/uncertainty-%j.out  # Output file
#SBATCH --error=/scratch/yli44/logs/uncertainty-%j.err   # Error file

## Slurm can send you updates via email
#SBATCH --mail-type=BEGIN,END,FAIL         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..
#SBATCH --mail-user=yli44@gmu.edu     # Put your GMU email address here

## Specify how much memory your job needs. (2G is the default)
#SBATCH --mem=10G        # Total memory needed per task (units: K,M,G,T)

#SBATCH --gres=gpu:1

## Load the relevant modules needed for the job
module load python/3.6.7
module load cuda/10.1
source /scratch/yli44/anomaly_env/bin/activate

## Run your program or script
#python train_sseg_head.py
python train_sseg_head_dropout.py
#python train_sseg_head_duq.py
