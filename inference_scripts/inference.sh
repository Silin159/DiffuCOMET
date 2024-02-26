#!/bin/bash

BENCHMARK_Train=$1  # comfact_roc | webnlg
BENCHMARK_Eval=$2  # comfact_roc | comfact_persona | comfact_mutual | comfact_movie | webnlg
TGT=$3  # facts | heads | tails
MODEL_SIZE=$4  # base | large
Test_Set=$5  # test (for generating facts, heads) or test_${MODEL_SIZE}_${Train_Step_Head} (for generating tails in pipeline)
Train_Step=$6
SCHEDULE=$7
CHECKPOINT="ckpts/diffusion/${BENCHMARK_Train}_${TGT}_diffusion_${MODEL_SIZE}"
MODEL_NAME="ema_0.9999_"${Train_Step}".pt"
SCHEDULE_PATH="alpha_cumprod_step_"${SCHEDULE}".npy"
VAL_TXT="data/${BENCHMARK_Eval}_${TGT}/${Test_Set}"
OUT_DIR="results/${BENCHMARK_Eval}_${TGT}_${MODEL_SIZE}_${Train_Step}"
SEED=10708

GEN_BY_Q="False"
GEN_BY_MIX="False"
MIX_PROB=0
MIX_PART=1
TOP_P=-1
CLAMP="no_clamp"
BATCH_SIZE=10
SEQ_LEN_SRC=$8  # max context length, change to 256 for ComFact movie summary portion, maintain 128 for others
DIFFUSION_STEPS=2000
NUM_SAMPLES=-1
NOISE_AMPLIFY=1
FG_DO_SAMPLE="True"
FG_TOP_K=100
FG_TOP_P=0.9
FG_INPUT="pred_xstart"  # "pred_xstart" or "ground_xstart"

export TOKENIZERS_PARALLELISM=False
export CUDA_VISIBLE_DEVICES=0
python -u inference_main.py --model_name_or_path ${CHECKPOINT}/${MODEL_NAME} --sequence_len_src ${SEQ_LEN_SRC} \
--batch_size ${BATCH_SIZE} --num_samples ${NUM_SAMPLES} --top_p ${TOP_P} \
--time_schedule_path ${CHECKPOINT}/${SCHEDULE_PATH} --seed ${SEED} --val_txt_path ${VAL_TXT} \
--generate_by_q ${GEN_BY_Q} --generate_by_mix ${GEN_BY_MIX} \
--out_dir ${OUT_DIR} --diffusion_steps ${DIFFUSION_STEPS} --clamp ${CLAMP} \
--generate_by_mix_prob ${MIX_PROB} --generate_by_mix_part ${MIX_PART} \
--noise_amplifier $NOISE_AMPLIFY --fg_do_sample ${FG_DO_SAMPLE} \
--fg_top_k ${FG_TOP_K} --fg_top_p ${FG_TOP_P} \
--fg_input ${FG_INPUT}
