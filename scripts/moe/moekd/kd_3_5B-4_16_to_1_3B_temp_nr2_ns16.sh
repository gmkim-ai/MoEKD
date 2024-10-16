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
CKPT_NAME="sft_init_1_3B"
CKPT="${BASE_PATH}/results/moe/train/sft/sft_1_3B/e10-bs8-lr1e-05-G1-N2-NN1/sft_init"
# CKPT="huggyllama/llama-7b"
TEACHER_CKPT_NAME="3_5B-4_16"
TEACHER_CKPT="${BASE_PATH}/results/moe/train/sft/sft_3_5B-4_16/e10-bs4-lr1e-05-G1-N4-NN1/best_rougeL"
# MP_SIZE=4
# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/moe/"
# hp
BATCH_SIZE=4
LR=0.00001
GRAD_ACC=1
EVAL_BATCH_SIZE=32
# length
MAX_LENGTH=512
# runtime
SAVE_PATH="${BASE_PATH}/results/moe/train/moekd/moekd_1_3B"
# seed
SEED=10

# MoE KD
NUM_SELECTS=16
NUM_REPEATS=2


OPTS=""
# moekd
OPTS+=" --num-selects ${NUM_SELECTS}"
OPTS+=" --num-repeats ${NUM_REPEATS}"
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-type moe"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type llama"
# OPTS+=" --gradient-checkpointing"
# OPTS+=" --model-parallel"
# OPTS+=" --model-parallel-size ${MP_SIZE}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 4"
OPTS+=" --dev-num -1"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 10" #10
OPTS+=" --kd-ratio 0.5"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 4"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# type
OPTS+=" --type moekd"
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
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
while ! test -f ./results/moe/train/moekd/moekd_1_3B/e10-bs4-lr1e-05-G1-N4-NN1-kd0.5-topk${NUM_SELECTS}-nr${NUM_REPEATS}/best_rougeL/log.txt
do
    ${CMD}
    sleep 20
done

bash scripts/moe/eval/run_eval.sh . results/moe/train/moekd/moekd_1_3B/e10-bs4-lr1e-05-G1-N4-NN1-kd0.5-topk${NUM_SELECTS}-nr${NUM_REPEATS}/best_rougeL 15035 llama ${GPUS_PER_NODE}