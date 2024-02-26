#!/bin/bash

lm="bart-large"
benchmark=$1  # comfact_roc | webnlg
task="rel_pred"
eval_set="test"
gen_mode="greedy"
num_gpus=4
visible=0,1,2,3

# the best model: checkpoint-428580 for comfact_roc, checkpoint-124450 for webnlg
CUDA_VISIBLE_DEVICES=${visible} python -m torch.distributed.launch \
       --nproc_per_node ${num_gpus} BART_Rel_Pred/baseline.py \
       --params_file BART_Rel_Pred/baseline/configs/params-${lm}.json \
       --dataroot BART_Rel_Pred/data_rp/${benchmark}/${task} \
       --exp_name ${benchmark}-${lm}-${task}-${eval_set}-${gen_mode} \
       --eval_dataset ${eval_set} \
       --gen_mode ${gen_mode}
