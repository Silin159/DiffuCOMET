# DiffuCOMET

This is the source code for paper [DiffuCOMET: Contextual Commonsense Knowledge Diffusion](https://arxiv.org/abs/2402.17011).

Part of our code is modified from [SeqDiffuSeq](https://github.com/Yuanhy1997/SeqDiffuSeq) repository.

## Getting Started 

Create a **python 3.8** Conda environment and install the following packages:
```
conda install mpi4py
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Preparing Datasets and Toolkits

Our preprocessed datasets can be downloaded from [this link](https://drive.google.com/file/d/1DIbF0WxscgEPKv4mX00wypWeVt39LMWQ/view?usp=sharing), please place ``data/`` under this root directory, and ``data_rp/`` under the ``BART_Rel_Pred/`` directory.

Please also download our commonsense fact linking toolkit (ComFact_Linker) from [this link](https://drive.google.com/file/d/1BDwh1ZQZXWXw3gduSlICB81xOelBhHb0/view?usp=sharing), and place ``ComFact_Linker/`` under this root directory.

## Training

**DiffuCOMET-Fact seeded with BART-{base | large} models**:
```
# Training fact embedding module
# on ComFact benchmark knowledge (ATOMIC 2020):
bash ./train_scripts/train_embedding_{base|large}.sh comfact facts 32
# on WebNLG+ 2020 benchmark knowledge:
bash ./train_scripts/train_embedding_{base|large}.sh webnlg facts 64

# Training fact diffusion module
# on ComFact benchmark (ROCStories portion):
bash ./train_scripts/train_diffusion_{base|large}.sh comfact_roc facts 32 32 comfact_facts
# on WebNLG+ 2020 benchmark:
bash ./train_scripts/train_diffusion_{base|large}.sh webnlg facts 8 64 webnlg_facts
```

**DiffuCOMET-Entity seeded with BART-{base | large} models**:
```
# Training entity embedding module
# on ComFact benchmark knowledge (ATOMIC 2020):
bash ./train_scripts/train_embedding_{base|large}.sh comfact entities 32
# on WebNLG+ 2020 benchmark knowledge:
bash ./train_scripts/train_embedding_{base|large}.sh webnlg entities 64

# Training head entity diffusion module
# on ComFact benchmark (ROCStories portion):
bash ./train_scripts/train_diffusion_{base|large}.sh comfact_roc heads 16 16 comfact_entities
# on WebNLG+ 2020 benchmark:
bash ./train_scripts/train_diffusion_{base|large}.sh webnlg heads 8 16 webnlg_entities

# Training tail entity diffusion module
# on ComFact benchmark (ROCStories portion):
bash ./train_scripts/train_diffusion_{base|large}.sh comfact_roc tails 8 24 comfact_entities
# on WebNLG+ 2020 benchmark:
bash ./train_scripts/train_diffusion_{base|large}.sh webnlg tails 8 64 webnlg_entities

# Training relation prediction module
# on ComFact benchmark (ROCStories portion):
bash ./BART_Rel_Pred/train_rel_pred.sh comfact_roc
# on WebNLG+ 2020 benchmark:
bash ./BART_Rel_Pred/train_rel_pred.sh webnlg
```

**Model Checkpoints**

Our trained model checkpoints can be downloaded from [this link](https://drive.google.com/file/d/1i8WQ5TllcPylpcyCJqYEHpA2jx1Y1RZx/view?usp=sharing)


## Inference

**DiffuCOMET-Fact seeded with BART-{base | large} models**:
```
# Testing on ComFact ROCStories (comfact_roc), PersonaChat (comfact_persona), MuTual (comfact_mutual), MovieSummaries (comfact_movie) or WebNLG+ 2020 (webnlg):
bash ./inference_scripts/inference.sh ${train_dataset} ${test_dataset} facts {base|large} test ${train_step} ${schedule} ${ctx_len}

# ${train_dataset}: {comfact_roc|webnlg}
# ${test_dataset}: {comfact_roc|comfact_persona|comfact_mutual|comfact_movie|webnlg}
# ${train_step}: training step (ID) of tested model checkpoint, e.g., 130000
# ${schedule}: noise schedule ID of tested model checkpoint, should be ${train_step}-2000, e.g., 128000
# ${ctx_len}: maximum narrative context length, should be 256 for testing on comfact_movie, while 128 for others
# generations will be saved in results/${test_dataset}_facts_{base|large}_${train_step}/generations.json

# Post-processing fact generations, post-processed generations will be saved in ${result_dir}/gen_processed.json:
python ./diffu_eval/post_process_facts.py --context_dir data/${test_dataset}_facts/test.contexts \
    --result_dir results/${test_dataset}_facts_{base|large}_${train_step}
```

**DiffuCOMET-Entity seeded with BART-{base | large} models**:
```
# Head entity generation
bash ./inference_scripts/inference.sh ${train_dataset} ${test_dataset} heads {base|large} test ${train_step_head} ${schedule_head} ${ctx_len}

# Post-processing head entity generations:
python ./diffu_eval/post_process_heads.py --context data/${test_dataset}_heads/test.contexts \
    --result_dir results/${test_dataset}_heads_{base|large}_${train_step_head} \
    --tail_gen_input_dir data/${test_dataset}_tails/test_{base|large}_${train_step_head}

# Tail entity generation
bash ./inference_scripts/inference.sh ${train_dataset} ${test_dataset} tails {base|large} test_{base|large}_${train_step_head} ${train_step_tail} ${schedule_tail} ${ctx_len}

# Post-processing tail entity generations:
python ./diffu_eval/post_process_tails.py --gold_dir data/${test_dataset}_facts/test \
    --tail_gen_input_dir data/${test_dataset}_tails/test_{base|large}_${train_step_head} \
    --tail_gen_result_dir results/${test_dataset}_tails_{base|large}_${train_step_tail} \
    --pipeline_result_dir results/${test_dataset}_pipeline_{base|large}_${train_step_head}_${train_step_tail} \
    --rel_pred_input_dir BART_Rel_Pred/data_rp/${test_dataset}/rel_pred_inf_{base|large}/test
    
# Relation prediction
bash ./BART_Rel_Pred/run_rel_pred.sh ${train_dataset} ${test_dataset} ${train_step_rel_pred} {base|large}

# Post-processing relation predictions:
python ./diffu_eval/post_process_rel_pred.py --test_data ${test_dataset} \
    --rel_pred_ids BART_Rel_Pred/data_rp/${test_dataset}/rel_pred_inf_{base|large}/test/labels.json \
    --rel_pred_results BART_Rel_Pred/pred/${test_dataset}-{base|large}/predictions.json \
    --pipeline_result_dir results/${test_dataset}_pipeline_{base|large}_${train_step_head}_${train_step_tail}
```

## Evaluation

**Evaluating on traditional NLG metrics**
```
python ./diffu_eval/eval_nlg.py --generation ${processed_gen} --eval_result_dir ${eval_result_dir}

# ${processed_gen}: results/${test_dataset}_facts_{base|large}_${train_step}/gen_processed.json
# or results/${test_dataset}_pipeline_{base|large}_${train_step_head}_${train_step_tail}/gen_processed.json

# ${eval_result_dir}: directory for saving evaluation scores, e.g., results/${test_dataset}_facts_{base|large}_${train_step}
# evaluation scores will be saved in ${eval_result_dir}/nlg_eval.json
```

**Evaluating on our proposed clustering-based metrics**
```
# Pre-processing generations for ComFact linker to score relevance
python ./diffu_eval/prepare_comfact_linking.py --test_data ${test_dataset} --generation ${processed_gen} \
    --comfact_input_dir ComFact_Linker/data_fl/all/fact_link/nlu/${eval_model}

# ${eval_model}: ${test_dataset}_facts_{base|large}_${train_step}
# or ${test_dataset}_pipeline_{base|large}_${train_step_head}_${train_step_tail}

# Switching to ComFact original environment
# Please refer to ComFact_Linker/README.md

# Run ComFact linker
bash ComFact_Linker/run_fact_link.sh ${eval_model}
# scoring results will be saved in ComFact_Linker/pred/${eval_model}/predictions.json

# Switching back to DiffuCOMET environment
# Please refer to ComFact_Linker/README.md

# Post-processing ComFact linker scoring results:
python ./diffu_eval/write_comfact_scores.py --comfact_output ComFact_Linker/pred/${eval_model}/predictions.json \
    --generation ${processed_gen}

# Clustering-based evaluation
python ./diffu_eval/eval_cluster.py --test_data ${test_dataset} --generation ${processed_gen} \
    --eval_result_dir ${eval_result_dir}
    
# evaluation scores will be saved in ${eval_result_dir}/cluster_eval.csv
# each line of the CSV file records a metric scoring on a range of clustering thresholds (DBSCAN eps)
```

**Evaluating on WebNLG metrics (for testing dataset webnlg)**
```
python ./diffu_eval/eval_webnlg.py --generation ${processed_gen} --eval_result_dir ${eval_result_dir}

# evaluation scores will be saved in ${eval_result_dir}/scores_webnlg.json
```
