#!/bin/bash
set -u

LOSS_FUNC="uniform"
LR=0.00001  # learning rate
SRC="contexts"
BENCHMARK=$1  # comfact_roc | webnlg
TGT=$2  # facts | heads | tails
SEQ_LEN_SRC=128  # max context length
SEQ_LEN=$3  # max number of fact/entity, 32, 16 or 8 for fact, head or tail generation on ComFact, 8 for WebNLG
SEQ_LEN_FACT=$4  # max fact/entity length, 32, 16 or 24 for fact, head or tail generation on ComFact, 64, 16, 64 for fact, head or tail generation on WebNLG
EMB_PT_SOURCE=$5  # comfact_facts | comfact_entities | webnlg_facts | webnlg_entities, pretrained embedding module source
SCHEDULE_UPDATE=2000  # dynamic noise schedule updates every 2000 training steps
WARMUP=2000  # warmup steps
LR_ANNEAL_STEPS=200000  # total training steps, the best checkpoint is around 160k, 140k or 110k steps for fact, head or tail generation
DSET="diffusion"  # diffusion training
MICRO_BSZ=2
UPDATE_GRANU=20
INIT_PRETRAINED_MODEL="True"
INIT_PRETRAINED_EMBEDDER="True"
USE_PRETRAINED_EMBEDDINGS="False"
FREEZE_EMBEDDINGS="False"
USE_PRETRAINED_TOKENIZER="True"
DAE="False"  # denoise auto encoding, i.e., attention to word embeds
DIFFUSION_STEPS=2000  # total diffusion steps
NOISE_SCHEDULE="sqrt"  # initial noise schedule
BATCH_SIZE=2
CHECKPOINT_PATH="ckpts/${DSET}/${BENCHMARK}_${TGT}_diffusion_large"
TRAIN_TXT_PATH="./data/${BENCHMARK}_${TGT}/train"
VAL_TXT_PATH="./data/${BENCHMARK}_${TGT}/test"
CHANNELS=1024
INTER_CHANNELS=4096
LAYERS=12
WEIGHT_DECAY=0.0
SEED=10708
DROPOUT=0.1
NUM_HEADS=16
CONFIG_NAME_MODEL="facebook/bart-large"
CONFIG_NAME_EMBEDDER="ckpts/embedding/${EMB_PT_SOURCE}_emb_pt_large/model300000"
GAMMA_NLL=0.01  # gamma for weighting MSE and anchor losses used for diffusion training
NOISE_AMPLIFY=4  # factor for amplifying the standard deviation of diffusion noise
CONFIG_NAME_TOKENIZER="facebook/bart-large"
NOTES="diffusion training"

mkdir -p ${CHECKPOINT_PATH}
mkdir -p ${CHECKPOINT_PATH}/log/
export DIFFUSION_BLOB_LOGDIR=${CHECKPOINT_PATH}/log/

ARGS=(--checkpoint_path ${CHECKPOINT_PATH}
    --save_interval ${WARMUP} --lr ${LR}
    --batch_size ${BATCH_SIZE}
    --src ${SRC}
    --tgt ${TGT}
    --diffusion_steps ${DIFFUSION_STEPS}
    --noise_schedule ${NOISE_SCHEDULE}
    --sequence_len ${SEQ_LEN} --seed ${SEED}
    --weight_decay ${WEIGHT_DECAY}
    --predict_xstart True
    --train_txt_path ${TRAIN_TXT_PATH}
    --dataset "${BENCHMARK}_${TGT}"
    --val_txt_path ${VAL_TXT_PATH}
    --config_name ${CONFIG_NAME_MODEL}
    --config_name_embedder ${CONFIG_NAME_EMBEDDER}
    --init_pretrained ${INIT_PRETRAINED_MODEL}
    --init_pretrained_embedder ${INIT_PRETRAINED_EMBEDDER}
    --freeze_embeddings ${FREEZE_EMBEDDINGS}
    --use_pretrained_embeddings ${USE_PRETRAINED_EMBEDDINGS}
    --dae ${DAE}
    --gamma_nll ${GAMMA_NLL}
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
    --sequence_len_src $SEQ_LEN_SRC
    --sequence_len_fact $SEQ_LEN_FACT
    --warmup $WARMUP
    --schedule_sampler $LOSS_FUNC
    --pretrained_tokenizer $CONFIG_NAME_TOKENIZER
    --use_pretrained_tokenizer $USE_PRETRAINED_TOKENIZER
    --noise_amplifier $NOISE_AMPLIFY
    --microbatch $MICRO_BSZ
    --loss_update_granu $UPDATE_GRANU
    --schedule_update_stride $SCHEDULE_UPDATE)

NUM_GPUS=4
export TOKENIZERS_PARALLELISM=False
# export CUDA_VISIBLE_DEVICES=0,1,2,3 &&
mpiexec -n $NUM_GPUS python -u main.py "${ARGS[@]}"

