#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J Rumex_PSPNet
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 23:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=128GB]"
#BSUB -R "select[gpu32gb]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u ronjag@elektro.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o hpc_logs/gpu-%J.out
#BSUB -e hpc_logs/gpu_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/10.2
module load python3/3.6.7

# Install pip requirements
python3 -m pip install --user -r requirements.txt

/appl/cuda/10.2/samples/NVIDIA_CUDA-10.2_Samples/bin/x86_64/linux/release/deviceQuery

python3 train.py