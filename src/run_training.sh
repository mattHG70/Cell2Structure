#!/bin/bash
# Slurm script to run the transfer learning job.
# Uses one node + one gpu
# Slurm PARAMETERS
#SBATCH --job-name=team6_training
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32g
#SBATCH --account=siads699f24_class
#SBATCH --mail-user mhuebsch@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/siads699/%x-%j.log
#SBATCH --error=/home/%u/siads699/error-%x-%j.log

# run image embedding creation as a cluster job on Great Lakes HPC
# using 1 node and 1 gpu

# run bashrc
source ~/.bashrc
# switch to directory containing all scripts
cd ~/siads699/Cell2Structure/src
# load a python+conda module if necessary
# module load <ptyhon module>
module load cuda cudnn
# activate custom conda environment
mamba activate capstone

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

START=`date +%s`; STARTDATE=`date`;
echo [INFO] [$START] [$STARTDATE] [$$] [$JOB_ID] Starting the workflow
echo [INFO] [$START] [$STARTDATE] [$$] [$JOB_ID] We got the following cores: $CUDA_VISIBLE_DEVICES

# set device to "cuda" to enable gpu usage
python model_training.py -bit_depth 8 \
                         -n_epochs 20 \
                         -batch_size 64 \
                         -learning_rate 0.0001 \
                         -data_augment False \
                         -unfreeze True \
                         -model_type l2l \
                         -config ~/siads699/Cell2Structure/project_config.toml

EXITCODE=$?

END=`date +%s`; ENDDATE=`date`
echo [INFO] [$END] [$ENDDATE] [$$] [$JOB_ID] Workflow finished with code $EXITCODE
echo [INFO] [$END] [`date`] [$$] [$JOB_ID] Workflow execution time \(seconds\) : $(( $END-$START ))
