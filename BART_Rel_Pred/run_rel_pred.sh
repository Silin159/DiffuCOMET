#!/bin/bash

# bash run_rel_pred.sh comfact|webnlg entity_generation_model
lm="bart-large"
train_benchmark=$1  # comfact_roc|webnlg
test_benchmark=$2  # comfact_roc|comfact_persona|comfact_mutual|comfact_movie|webnlg
task="rel_pred"
train_step=$3  # model checkpoint ID
model=$4  # base | large
eval_set="test"
gen_mode="greedy"

mkdir -p BART_Rel_Pred/pred/${test_benchmark}-${model}
visible=0

CUDA_VISIBLE_DEVICES=${visible} python BART_Rel_Pred/baseline.py \
   --eval_only \
   --checkpoint BART_Rel_Pred/run/${train_benchmark}-${lm}-${task}-${eval_set}-${gen_mode}/checkpoint-${train_step} \
   --params_file BART_Rel_Pred/baseline/configs/params-${lm}.json \
   --eval_dataset ${eval_set} \
   --dataroot BART_Rel_Pred/data_rp/${test_benchmark}/rel_pred_inf_${model} \
   --output_file BART_Rel_Pred/pred/${test_benchmark}-${model}/predictions.json \
   --gen_mode ${gen_mode}
