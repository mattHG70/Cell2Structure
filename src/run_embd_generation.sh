#!/bin/bash

# Slurm script to run the image embedding generation job.
# Uses one node + one gpu
# Slurm PARAMETERS
#SBATCH --job-name=team6_embd_gen
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
# the image_path parameters contains the full path the images and need to be adapted
python generate_embeddings.py -dataset_path ../data/processed/BBBC021_final_original_dataset.csv \
                              -image_path /scratch/siads699f24_class_root/siads699f24_class/mhuebsch/images \
                              -model_path ../data_augment/models/capstone_model_32_25_0.0001_1_16.pt \
                              -output_path ../data/interim/embeddings/l2l_test_16bit.npy \
                              -batch_size 32 \
                              -bit_depth 16 \
                              -base_model False

EXITCODE=$?

END=`date +%s`; ENDDATE=`date`
echo [INFO] [$END] [$ENDDATE] [$$] [$JOB_ID] Workflow finished with code $EXITCODE
echo [INFO] [$END] [`date`] [$$] [$JOB_ID] Workflow execution time \(seconds\) : $(( $END-$START ))
