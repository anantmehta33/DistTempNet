#!/bin/bash

##NECESSARY JOB SPECIFICATIONS  
#SBATCH --job-name=TempNetTraining            # Set the job name to "TempNetTraining"
#SBATCH --time=40:00:00                        # Set the wall clock limit to 30 hours
#SBATCH --ntasks=8                           # Request 8 tasks (for distributed training)
#SBATCH --ntasks-per-node=2                    # Request 8 tasks per node
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G                              # Request 80GB of memory
#SBATCH --output=/scratch/group/optmai/anant/Bimodal_cc12m/ViTb32.%j  # Send stdout/err to "TempNetOut.[jobID]"
#SBATCH --gres=gpu:a100:2                    # Request 8 A100 GPUs (2 per node)
#SBATCH --partition=gpu                         # Specify the partition for GPU jobs

# Set environment variables
export TRANSFORMERS_OFFLINE=1
data_path=/scratch/group/optmai/datasets/cc12m_webdataset/cc12m # Base data path 
imagenet_val_path=/scratch/group/optmai/datasets/imagenet/val # Updated ImageNet path
train_image_root=/scratch/group/optmai/datasets/cc12m_webdataset/cc12m # New CC3M train path  
data=cc12m
train_file=/scratch/user/anant_mehta/TempNet/clip_train/cc12m_full_data.json # Updated train_file path
val_coco_file=/scratch/user/anant_mehta/TempNet/clip_train/coco_val.json  # Add this line for the validation file
lr=0.0008
frac=1.0
desc=isogclr_tempnet
gamma=0.8
rho=9.5

# Change to the directory containing your script
cd /scratch/group/optmai/anant/Bimodal_cc12m
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=4820




# Run your training script with the appropriate parameters
srun python -u  clip_cc12m.py \
    --train-data '/scratch/group/optmai/datasets/cc12m_webdataset/cc12m/{00000..01242}.tar' \
    --train-num-samples 9184307 --data_size 12423374 \
    --dataset_type webdataset\
    --data_path ${data_path} \
    --data ${data} \
    --train_file ${train_file} \
    --train_image_root ${train_image_root} \
    --val_coco_file ${val_coco_file} \
    --output_dir output/TempNet_ViT_${data}_gamma${gamma}_rho${rho}_${desc} \
    --use_amp \
    --init_model \
    --image_encoder ViT \
    --text_encoder Transformer \
    --epochs 33 --lr ${lr} \
    --lr_temp_net 3e-5 \
    --rho ${rho} \
    --train_frac ${frac} \
    --zs_dataset imagenet \
    --zs_datafolder ${imagenet_val_path} \
    --ita_type isogclr_tempnet \
    --batch_size_train 512 \
    --world_size 8 \
    --distributed True \
    --warmup_epochs 20 \
    --workers 2 \
    --seed 2024 \
    --image_res 224 \
    --sogclr_gamma ${gamma} \
    --embed_dim 512

