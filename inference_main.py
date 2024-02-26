"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os, json
from typing import List
import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from src.utils import dist_util, logger
from src.modeling.diffusion.gaussian_diffusion import _extract_into_tensor

from model_utils import create_model_and_diffusion
from args_utils import create_argparser, args_to_dict, model_and_diffusion_defaults
from tokenizer_utils import create_tokenizer
import dataloader_utils
from mpi4py import MPI

ADD_TOKENS_VALUES = ["<utter_sep>", "<past>", "<center>", "<future>",
                     "personx", "persony", "personz", "<eos_fact>",
                     "<atlocation>", "<capableof>", "<causes>", "<desires>",
                     "<hasproperty>", "<hassubevent>", "<hinderedby>", "<isafter>",
                     "<isbefore>", "<madeupof>", "<notdesires>", "<objectuse>",
                     "<oeffect>", "<oreact>", "<owant>", "<xattr>", "<xeffect>",
                     "<xintent>", "<xneed>", "<xreact>", "<xreason>", "<xwant>"]


def main():

    args = create_argparser().parse_args()
    print(args.fg_input)
    set_seed(args.seed)
    th.manual_seed(args.seed)
    print(args.seed)
    dist_util.setup_dist()
    logger.configure()

    # load configurations.
    args.checkpoint_path = os.path.split(args.model_name_or_path)[0]

    config_path = os.path.join(args.checkpoint_path, "training_args.json")
    training_args = read_training_args(config_path)
    training_args["batch_size"] = args.batch_size
    training_args["diffusion_steps"] = args.diffusion_steps
    training_args['model_name_or_path'] = args.model_name_or_path
    training_args["clamp"] = args.clamp
    training_args['out_dir'] = args.out_dir
    training_args['num_samples'] = args.num_samples
    training_args['val_txt_path'] = args.val_txt_path
    training_args['top_p'] = args.top_p
    training_args['sequence_len_src'] = args.sequence_len_src
    # training_args['sequence_len_fact'] = args.sequence_len_fact
    # training_args['sequence_len'] = args.sequence_len
    training_args['generate_by_q'] = args.generate_by_q
    training_args['generate_by_mix'] = args.generate_by_mix
    training_args['generate_by_mix_prob'] = args.generate_by_mix_prob
    training_args['generate_by_mix_part'] = args.generate_by_mix_part
    training_args['time_schedule_path'] = args.time_schedule_path
    training_args['noise_amplifier'] = args.noise_amplifier
    training_args['fg_do_sample'] = args.fg_do_sample
    fg_max_len = training_args['sequence_len']
    training_args['fg_max_len'] = fg_max_len
    training_args['fg_top_k'] = args.fg_top_k
    training_args['fg_top_p'] = args.fg_top_p
    training_args['fg_input'] = args.fg_input
    training_args['seed'] = args.seed
    
    args.__dict__.update(training_args)
    args.sigma_small = True
    print(args.fg_input)

    logger.info(f"Init pretrained = {args.init_pretrained}")
    logger.info(f"Freeze embeddings = {args.freeze_embeddings}")
    logger.info(f"Use pretrained embeddings = {args.use_pretrained_embeddings}")
    logger.info(f"Use pretrained embeddings = {args.use_pretrained_tokenizer}")
    
    tokenizer = create_tokenizer(return_pretokenized=args.use_pretrained_tokenizer,
                                 path=f"data/{args.dataset}/",
                                 tokenizer_type='byte-level',
                                 tokenizer_ckpt=args.pretrained_tokenizer)
    # add special tokens
    tokenizer.add_tokens(ADD_TOKENS_VALUES)
    model_args_dict = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(
        tokenizer=tokenizer,
        pad_tok_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else tokenizer.get_vocab()['<pad>'],
        resume_checkpoint=args.resume_checkpoint, embedder_args=model_args_dict, **model_args_dict
    )

    diffusion._load_time_schedule(args.time_schedule_path)
    model.load_state_dict(dist_util.load_state_dict(args.model_name_or_path, map_location="cpu"))
    model.eval()

    print('data path', args.val_txt_path)
    val_dataloader = dataloader_utils.get_dataloader_kg(
        tokenizer=tokenizer,
        args=args,
        data_path=args.val_txt_path,
        batch_size=args.batch_size,
        max_seq_len_src=args.sequence_len_src,
        max_seq_len=args.sequence_len,
        max_fact_len=args.sequence_len_fact
    )

    if args.num_samples <= 0:
        args.num_samples = len(dataloader_utils.TextDatasetSeq2KG(tokenizer=tokenizer, data_path=args.val_txt_path,
                                                                  source=args.src, target=args.tgt,
                                                                  shard=MPI.COMM_WORLD.Get_rank(),
                                                                  num_shards=MPI.COMM_WORLD.Get_size()))
        logger.log(f"sample count is {args.num_samples}")
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"the parameter count is {pytorch_total_params}")

    diffusion.rescale_timesteps = True

    model.to(dist_util.dev())
    model.eval()  # DEBUG

    logger.log("sampling...")
    logger.log(f"Clamping is set to {args.clamp}")
    all_samples = []
    ground_true_samples = []
    while len(all_samples) < args.num_samples:
        batch, _ = next(val_dataloader)
        model_kwargs = {key: item.to(dist_util.dev()) for key, item in batch.items()
                        if 'decoder' not in key and 'embedder' not in key}
        sample_shape = (args.batch_size, args.sequence_len, model.input_transformers.shared.weight.shape[1])
        print('sample_shape', sample_shape)

        # return key "pred_xstart" or "greedy_mean"
        sample = diffusion.p_sample_loop(
                model,
                sample_shape,
                clip_denoised=args.clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
                top_p=args.top_p,
                progress=True,
                generate_by_q=args.generate_by_q,
                generate_by_mix=args.generate_by_mix,
                generate_by_mix_prob=args.generate_by_mix_prob,
                generate_by_mix_part=args.generate_by_mix_part,
                return_key="greedy_mean"
            )

        embedder_input_ids = batch["embedder_enc_input_ids"].to(dist_util.dev()).contiguous()
        first_shape = embedder_input_ids.shape[0] * embedder_input_ids.shape[1]
        embedder_attention_mask = batch["embedder_attention_mask"].to(dist_util.dev()).contiguous()
        fact_embeds = model.get_fact_embeds(embedder_input_ids, embedder_attention_mask)
        # modified, fact_out_embeds can be: sample, fact_embeds[:, :, 0, :] (cheating)
        if args.fg_input == "ground_xstart":
            fact_out_embeds = fact_embeds[:, :, 0, :]
        elif args.fg_input == "noisy_xstart":
            x_start_mean = fact_embeds[:, :, 0, :]
            std = _extract_into_tensor(
                diffusion.sqrt_one_minus_alphas_cumprod,
                th.tensor([0]).to(x_start_mean.device),
                x_start_mean.shape,
            )
            fact_out_embeds = diffusion.get_x_start(x_start_mean, std)
        else:
            fact_out_embeds = sample

        if args.dae:
            input_fact_embeds = th.cat([fact_out_embeds.unsqueeze(2), fact_embeds[:, :, 1:, :]], dim=2).contiguous()
            # cross_attention_mask = shift_right_3d(embedder_attention_mask, 0, 1).contiguous()
            cross_attention_mask = embedder_attention_mask.contiguous()
        else:
            input_fact_embeds = fact_out_embeds.unsqueeze(2).contiguous()
            cross_attention_mask = None
        de_embed_input_ids = th.ones(first_shape, 1).long() * model.embedders.config.decoder_start_token_id
        de_embed_input_ids = de_embed_input_ids.to(dist_util.dev()).contiguous()

        gen_facts = model.generate_facts(input_fact_embeds=input_fact_embeds,
                                         de_embed_input_ids=de_embed_input_ids,
                                         cross_attention_mask=cross_attention_mask,
                                         gen_args=args, tokenizer=tokenizer)  # bsz, seqlen, factlen, vocab

        # cands = th.topk(logits, k=1, dim=-1).indices.squeeze()
        # if args.decoder_attention_mask:
        #     cands[model_kwargs['decoder_attention_mask']==0] = 1
        # gathered_samples = [th.zeros_like(gen_facts) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, gen_facts)  # gather not supported with NCCL
        all_samples.extend(gen_facts)

        ground_ids = batch['embedder_input_ids']
        ground_facts = []
        batch_size = input_fact_embeds.shape[0]
        for batch in range(batch_size):
            ground_facts_batch = tokenizer.batch_decode(ground_ids[batch, :, :], skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=True)
            ground_facts_batch = list(map(str.strip, ground_facts_batch))
            ground_facts.append(ground_facts_batch)
        # gathered_ground_trues = [th.zeros_like(batch['decoder_input_ids']) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_ground_trues, batch['decoder_input_ids'])
        ground_true_samples.extend(ground_facts)

        logger.log(f"tested {len(all_samples)} samples")

    '''
    cands = np.concatenate(all_samples, axis=0)
    cands = cands[: args.num_samples]

    decoded_sentences = []
    for seq in cands:
        seq = seq[seq>2]
        decoded_sentence = tokenizer.decode(seq.tolist(), skip_special_tokens=True)
        decoded_sentences.append(decoded_sentence)
    
    ground_true_sentences = []
    ground_true_samples = np.concatenate(ground_true_samples, axis=0)[: args.num_samples]
    for seq in ground_true_samples:
        seq = seq[seq>2]
        ground_true_sentence = tokenizer.decode(seq.squeeze().tolist(), skip_special_tokens=True)
        ground_true_sentences.append(ground_true_sentence)

    dist.barrier()
    '''
    logger.log("sampling complete")
    all_samples = all_samples[:args.num_samples]
    ground_true_samples = ground_true_samples[:args.num_samples]

    write_outputs_facts(args=args, gen_facts=all_samples, gt_facts=ground_true_samples)


def load_embeddings(checkpoint_path, tokenizer, emb_dim):
    embeddings = th.nn.Embedding(tokenizer.vocab_size, emb_dim)
    embeddings.load_state_dict(th.load(f'{checkpoint_path}/random_emb.torch'))
    return embeddings


def read_training_args(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def write_outputs(args: dict, sentences: List[str], gt_sentences: List[str], raw_sentences, raw_gt_sentences) -> None:

    model_dir = os.path.split(args.model_name_or_path)[0]
    model_base_name = os.path.split(args.model_name_or_path)[1]
    if args.generate_by_q:
        comments = f'predict_by_qsample_{args.seed}'
    elif args.generate_by_mix:
        comments = f'predict_by_mixsample_{args.generate_by_mix_prob}_{args.generate_by_mix_part}_{args.seed}'
    else:
        comments = f'normal_{args.seed}'
    num_samples = len(sentences)
    output_file_basepath = os.path.join(
        model_dir,
        f"{model_base_name}.samples_{num_samples}.steps-{args.diffusion_steps}.clamp-{args.clamp}-{comments}",
    ) + ".txt"
    with open(output_file_basepath, "w") as text_fout:
        for generated_sentence, ground_true_sentence in zip(sentences, gt_sentences):
            text_fout.write(json.dumps([generated_sentence, ground_true_sentence]) + "\n")

        print(f"written the decoded output to {output_file_basepath}")

    output_file_basepath = os.path.join(
        model_dir,
        f"{model_base_name}.samples_{num_samples}.steps-{args.diffusion_steps}.clamp-{args.clamp}.raw-output-ids-{comments}",
    ) + ".txt"
    with open(output_file_basepath, "w") as text_fout:
        for generated_sentence, ground_true_sentence in zip(raw_sentences, raw_gt_sentences):
            text_fout.write(json.dumps([generated_sentence.tolist(), ground_true_sentence.tolist()]) + "\n")

        print(f"written the decoded output to {output_file_basepath}")


def write_outputs_facts(args, gen_facts: List[List[str]], gt_facts: List[List[str]]) -> None:

    # model_dir = os.path.split(args.model_name_or_path)[0]
    # model_base_name = os.path.split(args.model_name_or_path)[1]
    # model_dir = os.path.split(args.out_dir)[0]
    # portion_name = os.path.split(args.out_dir)[1]
    # num_samples = len(gen_facts)
    # output_file_basepath = os.path.join(
    #     model_dir,
    #     f"{portion_name}.samples-{num_samples}.steps-{args.diffusion_steps}.seed-{args.seed}",
    # ) + ".json"
    results = []
    for generated_kg, ground_true_kg in zip(gen_facts, gt_facts):
        results.append({"gen": generated_kg, "ground": ground_true_kg})

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.out_dir+"/generations.json", "w") as text_fout:
        json.dump(results, text_fout, indent=2)
    print(f"written the model output to {args.out_dir}")


if __name__ == "__main__":
    main()
