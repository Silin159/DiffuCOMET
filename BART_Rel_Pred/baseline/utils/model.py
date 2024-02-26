import torch
# import torch.nn.functional as F
import logging
import numpy as np

logger = logging.getLogger(__name__)


def softmax(array, axis=1):
    e_x = np.exp(array)
    sum_axis = np.sum(e_x, axis=axis, keepdims=True)
    return e_x / sum_axis


def run_batch_linking(args, model, batch, tokenizer=None):
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, mc_token_ids, lm_labels, labels = batch

    model_outputs = model(input_ids=input_ids, labels=labels)

    cls_loss = model_outputs[0]
    cls_logits = model_outputs[1]
    lm_logits = None

    return cls_loss, lm_logits, cls_logits, labels


def run_batch_generation_train(args, model, batch, tokenizer=None):
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, decoder_input_ids, lm_label_ids = batch

    outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=lm_label_ids)

    gen_loss = outputs[0]

    return gen_loss


def run_batch_generation_eval(args, model, batch, tokenizer):
    batch_t = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids = batch_t

    batch_ut = tuple(input_tensor for input_tensor in batch if not isinstance(input_tensor, torch.Tensor))
    lm_target_text, data_info = batch_ut

    batch_size = input_ids.size(0)

    if args.gen_mode == "greedy":
        do_sample = False
        num_beams = 1
        num_beam_groups = 1
        diversity_penalty = 0.0
        k_sample = 0
        p_sample = 0.0
        num_seq = 1
    elif args.gen_mode == "sample_5":
        do_sample = True
        num_beams = 1
        num_beam_groups = 1
        diversity_penalty = 0.0
        k_sample = 100
        p_sample = 0.95
        num_seq = 5
    elif args.gen_mode == "sample_10":
        do_sample = True
        num_beams = 1
        num_beam_groups = 1
        diversity_penalty = 0.0
        k_sample = 100
        p_sample = 0.95
        num_seq = 10
    elif args.gen_mode == "sample_15":
        do_sample = True
        num_beams = 1
        num_beam_groups = 1
        diversity_penalty = 0.0
        k_sample = 100
        p_sample = 0.95
        num_seq = 15
    elif args.gen_mode == "beam_diverse_5":
        do_sample = False
        num_beams = 5
        num_beam_groups = 5
        diversity_penalty = 1.0
        k_sample = 0
        p_sample = 0.0
        num_seq = 5
    elif args.gen_mode == "beam_diverse_10":
        do_sample = False
        num_beams = 10
        num_beam_groups = 5
        diversity_penalty = 1.0
        k_sample = 0
        p_sample = 0.0
        num_seq = 10
    elif args.gen_mode == "beam_diverse_15":
        do_sample = False
        num_beams = 15
        num_beam_groups = 5
        diversity_penalty = 1.0
        k_sample = 0
        p_sample = 0.0
        num_seq = 15
    else:
        raise ValueError

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):  # validation
        gen_ids = model.module.generate(input_ids=input_ids, use_cache=True, num_beams=num_beams, do_sample=do_sample,
                                        top_k=k_sample, top_p=p_sample, num_beam_groups=num_beam_groups,
                                        diversity_penalty=diversity_penalty, num_return_sequences=num_seq)
    else:  # test
        gen_ids = model.generate(input_ids=input_ids, use_cache=True, num_beams=num_beams, do_sample=do_sample,
                                 top_k=k_sample, top_p=p_sample, num_beam_groups=num_beam_groups,
                                 diversity_penalty=diversity_penalty, num_return_sequences=num_seq)

    gen_text = []
    for split in range(0, batch_size * num_seq, num_seq):
        gen_text_single = tokenizer.batch_decode(gen_ids[split:(split+num_seq), :], skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=True)
        gen_text_single = list(map(str.strip, gen_text_single))
        gen_text.append(gen_text_single)

    return gen_text, lm_target_text, data_info
