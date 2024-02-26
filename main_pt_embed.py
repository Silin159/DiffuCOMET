"""
Train a diffusion model on images.
"""

import json
import pathlib
from transformers import set_seed
import os

from src.utils import dist_util, logger
from model_utils import create_embedder
from trainer import Trainer_PTE
import dataloader_utils
from args_utils import create_argparser, args_to_dict, model_and_diffusion_defaults
from tokenizer_utils import create_tokenizer

SPECIAL_TOKENS = {
    "additional_special_tokens":
                    ["<utter_sep>", "<past>", "<center>", "<future>", "<rel_bos>", "<rel_eos>",
                     "personx", "persony", "personz", "<fact_sep>", "<eos_fact>",
                     "<atlocation>", "<capableof>", "<causes>", "<desires>",
                     "<hasproperty>", "<hassubevent>", "<hinderedby>", "<isafter>",
                     "<isbefore>", "<madeupof>", "<notdesires>", "<objectuse>",
                     "<oeffect>", "<oreact>", "<owant>", "<xattr>", "<xeffect>",
                     "<xintent>", "<xneed>", "<xreact>", "<xreason>", "<xwant>"]
}

ADD_TOKENS_VALUES = ["<utter_sep>", "<past>", "<center>", "<future>", "<rel_bos>", "<rel_eos>",
                     "personx", "persony", "personz", "<fact_sep>", "<eos_fact>",
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
    # tokenizer.add_tokens(ADD_TOKENS_VALUES)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    train_dataloader = dataloader_utils.get_dataloader_pte(
        tokenizer=tokenizer,
        args=args,
        data_path=args.train_txt_path,
        batch_size=args.batch_size,
        max_fact_len=args.sequence_len_fact
    )

    val_dataloader = dataloader_utils.get_dataloader_pte(
        tokenizer=tokenizer,
        args=args,
        data_path=args.val_txt_path,
        batch_size=args.batch_size,
        max_fact_len=args.sequence_len_fact
    )

    args.vocab_size = len(tokenizer)  # tokenizer.vocab_size
    
    logger.log("creating embedding model...", args.checkpoint_path)
    model = create_embedder(tokenizer=tokenizer, **args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.to(dist_util.dev()) 
    
    print(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f"the parameter count is {pytorch_total_params}")

    logger.log(f"saving the hyperparameters to {args.checkpoint_path}/training_args.json")
    with open(f"{args.checkpoint_path}/training_args.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    logger.log("training...")
    Trainer_PTE(
        model=model,
        data=train_dataloader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=val_dataloader,
        eval_interval=args.eval_interval,
        warmup=args.warmup,
        dae=args.dae
    ).run_loop()


def make_tensorboard_name_from_args(args):
    keys_to_add = ["batch_size", "lr", "lr_anneal_steps", "config_name", "seed"]
    name = ""
    for key in keys_to_add:
        name += f"{key}={getattr(args, key)}_"
    return name


if __name__ == "__main__":
    main()
