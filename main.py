"""
Train a diffusion model on images.
"""

import json
import pathlib
import pprint
import sys
from transformers import set_seed
import os

from src.utils import dist_util, logger
from src.modeling.diffusion.resample import create_named_schedule_sampler
from model_utils import create_model_and_diffusion
from trainer import Trainer
import dataloader_utils
from args_utils import create_argparser, args_to_dict, model_and_diffusion_defaults
from tokenizer_utils import create_tokenizer

ADD_TOKENS_VALUES = ["<utter_sep>", "<past>", "<center>", "<future>",
                     "personx", "persony", "personz", "<eos_fact>",
                     "<atlocation>", "<capableof>", "<causes>", "<desires>",
                     "<hasproperty>", "<hassubevent>", "<hinderedby>", "<isafter>",
                     "<isbefore>", "<madeupof>", "<notdesires>", "<objectuse>",
                     "<oeffect>", "<oreact>", "<owant>", "<xattr>", "<xeffect>",
                     "<xintent>", "<xneed>", "<xreact>", "<xreason>", "<xwant>"]


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(dir=os.path.join(args.checkpoint_path, 'logger/'))
    set_seed(args.seed)
    print(f'set seed {args.seed + int(os.environ["RANK"])}')

    logger.log("creating data loader")
    pathlib.Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)

    tokenizer = create_tokenizer(return_pretokenized=args.use_pretrained_tokenizer,
                                 path=f"data/{args.dataset}/",
                                 tokenizer_type='byte-level',
                                 tokenizer_ckpt=args.pretrained_tokenizer)
    # add special tokens
    tokenizer.add_tokens(ADD_TOKENS_VALUES)

    train_dataloader = dataloader_utils.get_dataloader_kg(
        tokenizer=tokenizer,
        args=args,
        data_path=args.train_txt_path,
        batch_size=args.batch_size,
        max_seq_len_src=args.sequence_len_src,
        max_seq_len=args.sequence_len,
        max_fact_len=args.sequence_len_fact
    )

    val_dataloader = dataloader_utils.get_dataloader_kg(
        tokenizer=tokenizer,
        args=args,
        data_path=args.val_txt_path,
        batch_size=args.batch_size,
        max_seq_len_src=args.sequence_len_src,
        max_seq_len=args.sequence_len,
        max_fact_len=args.sequence_len_fact
    )
    args.vocab_size = len(tokenizer)  # tokenizer.vocab_size
    
    logger.log("creating model and diffusion...", args.checkpoint_path)
    model_args_dict = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(
        tokenizer=tokenizer,
        pad_tok_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else tokenizer.get_vocab()['<pad>'],
        resume_checkpoint=args.checkpoint_path, embedder_args=model_args_dict, **model_args_dict
    )
    model.to(dist_util.dev()) 
    
    print(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f"the parameter count is {pytorch_total_params}")
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f"saving the hyperparameters to {args.checkpoint_path}/training_args.json")
    with open(f"{args.checkpoint_path}/training_args.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    logger.log("training...")
    Trainer(
        model=model,
        diffusion=diffusion,
        data=train_dataloader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=val_dataloader,
        eval_interval=args.eval_interval,
        warmup=args.warmup,
        dae=args.dae,
        gamma_nll=args.gamma_nll
    ).run_loop()


def make_tensorboard_name_from_args(args):
    keys_to_add = ["batch_size", "lr", "lr_anneal_steps", "config_name", "seed"]
    name = ""
    for key in keys_to_add:
        name += f"{key}={getattr(args, key)}_"
    return name


if __name__ == "__main__":
    main()
