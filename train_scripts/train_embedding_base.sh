#!/bin/bash
set -u

LR=0.000002  # learning rate
WARMUP=3000   # warmup steps
SAVE=50000
LR_ANNEAL_STEPS=150000  # total training steps
DSET="embedding"
MICRO_BSZ=128
UPDATE_GRANU=20
INIT_PRETRAINED_EMBEDDER="True"
USE_PRETRAINED_EMBEDDINGS="False"
FREEZE_EMBEDDINGS="False"
USE_PRETRAINED_TOKENIZER="True"
DAE="False"
BATCH_SIZE=128
BENCHMARK=$1  # comfact | webnlg
TASK=$2  # facts | entities
SEQ_LEN_FACT=$3  # max fact/entity length, 32 or 64 for ComFact and WebNLG benchmarks
CHECKPOINT_PATH="ckpts/${DSET}/${BENCHMARK}_${TASK}_emb_pt_base"
TRAIN_TXT_PATH="./data/${BENCHMARK}_knowledge/${TASK}.json"
VAL_TXT_PATH="./data/${BENCHMARK}_knowledge/${TASK}.json"
CHANNELS=768
INTER_CHANNELS=3072
LAYERS=6
WEIGHT_DECAY=0.0
SEED=10708
DROPOUT=0.1
NUM_HEADS=12
CONFIG_NAME_EMBEDDER="facebook/bart-base"
CONFIG_NAME_TOKENIZER="facebook/bart-base"
NOTES="embedder pretraining"

mkdir -p ${CHECKPOINT_PATH}
mkdir -p ${CHECKPOINT_PATH}/log/
export DIFFUSION_BLOB_LOGDIR=${CHECKPOINT_PATH}/log/

ARGS=(--checkpoint_path ${CHECKPOINT_PATH}
    --save_interval ${SAVE} --lr ${LR}
    --batch_size ${BATCH_SIZE}
    --seed ${SEED}
    --weight_decay ${WEIGHT_DECAY}
    --predict_xstart True
    --train_txt_path ${TRAIN_TXT_PATH}
    --dataset "${BENCHMARK}_knowledge"
    --val_txt_path ${VAL_TXT_PATH}
    --config_name_embedder ${CONFIG_NAME_EMBEDDER}
    --init_pretrained_embedder ${INIT_PRETRAINED_EMBEDDER}
    --freeze_embeddings ${FREEZE_EMBEDDINGS}
    --use_pretrained_embeddings ${USE_PRETRAINED_EMBEDDINGS}
    --dae ${DAE}
    --notes \""${NOTES}"\")


if [ ${LR_ANNEAL_STEPS} -eq 0 ]; then
    LR_ANNEAL_STEPS=100
    DEBUG=true
else
    DEBUG=false
fi

ARGS+=(--lr_anneal_steps $LR_ANNEAL_STEPS)

if [ $DEBUG = true ]; then
    ARGS+=(--debug)
fi

ARGS+=(--encoder_layers $LAYERS
    --decoder_layers $LAYERS
    --num_heads $NUM_HEADS
    --in_channel $CHANNELS
    --out_channel $CHANNELS
    --num_channels $INTER_CHANNELS
    --sequence_len_fact $SEQ_LEN_FACT
    --warmup $WARMUP
    --pretrained_tokenizer $CONFIG_NAME_TOKENIZER
    --use_pretrained_tokenizer $USE_PRETRAINED_TOKENIZER
    --microbatch $MICRO_BSZ
    --loss_update_granu $UPDATE_GRANU)

NUM_GPUS=4
export TOKENIZERS_PARALLELISM=False
# export CUDA_VISIBLE_DEVICES=0,1,2,3 &&
mpiexec -n $NUM_GPUS python -u main_pt_embed.py "${ARGS[@]}"

