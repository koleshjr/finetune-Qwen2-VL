# Set visible GPUs
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Add debugging environment variables
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1

# Launch training with Accelerate
accelerate launch \
    --mixed_precision=no \
    --dynamo_backend=no \
    --num_machines=1 \
    --multi_gpu \
    --num_processes=4 \
    finetune_distributed.py
