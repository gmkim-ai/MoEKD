#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-4}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"."}
CKPT_NAME="3_5B-2_8"
CKPT="${BASE_PATH}/checkpoints/llama-moe/foundation/${CKPT_NAME}"
#CKPT="${BASE_PATH}/results/moe/train/sft/sft_3_0B-2_16/e10-bs1-lr5e-05-G8-N2-NN1-old-epoch8/best_rougeL"
# CKPT="huggyllama/llama-7b"
# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/moe/"
# hp
BATCH_SIZE=4
LR=0.00001 #0.0001
#MIN_LR=0.00001
GRAD_ACC=1
EVAL_BATCH_SIZE=32
# length
MAX_LENGTH=512
# runtime
SAVE_PATH="${BASE_PATH}/results/moe/train/sft/sft_3_5B-2_8"
# seed
SEED=10
SEED_ORDER=10


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type moe"
# OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 8"
OPTS+=" --dev-num -1"
# hp
OPTS+=" --lr ${LR}"
#OPTS+=" --lr-min ${MIN_LR}" #EDIT
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 0.1" #1e-2 EDIT
OPTS+=" --clip-grad 1.0" #1.0
OPTS+=" --epochs 10"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 10"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
OPTS+=" --seed-order ${SEED_ORDER}"
# # deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# type
OPTS+=" --type lm"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
export PT_HPU_LAZY_MODE=0
export OMP_NUM_THREADS=8
# OPTS+=" --use_lazy_mode=False"
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
while ! test -f ./results/moe/train/sft/sft_3_5B-2_8/e10-bs4-lr1e-05-G1-N4-NN1/6840/pytorch_model.bin
do
    ${CMD}
    sleep 20
done

#bash scripts/moe/eval/run_eval.sh . results/moe/train/sft/sft_3_5B-4_16/e10-bs4-lr1e-05-G1-N4-NN1/best_rougeL 15035 moe ${GPUS_PER_NODE}