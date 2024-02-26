import random
import logging

from itertools import chain
from copy import deepcopy

import torch

from tqdm import tqdm

from .utils.data import (
    pad_ids, truncate_sequences, truncate_sequences_dual
)

from .utils.dataset_walker import DatasetWalker

logger = logging.getLogger(__name__)

ADD_TOKENS_VALUES = ["<utter_sep>", "<past>", "<center>", "<future>", "<rel_bos>", "<rel_eos>",
                     "personx", "persony", "personz", "<fact_sep>", "<eos_fact>",
                     "<atlocation>", "<capableof>", "<causes>", "<desires>",
                     "<hasproperty>", "<hassubevent>", "<hinderedby>", "<isafter>",
                     "<isbefore>", "<madeupof>", "<notdesires>", "<objectuse>",
                     "<oeffect>", "<oreact>", "<owant>", "<xattr>", "<xeffect>",
                     "<xintent>", "<xneed>", "<xreact>", "<xreason>", "<xwant>"]


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        self.args = args
        self.dataroot = args.dataroot
        self.tokenizer = tokenizer
        self.split_type = split_type
        self.task_type = args.task

        self.dataset_walker = DatasetWalker(split_type, labels=labels, dataroot=self.dataroot, labels_file=labels_file)

        self.samples = self._prepare_samples()
        self._create_examples()

    def _prepare_samples(self):
        logger.info("Prepare fact generation samples")
        samples = []
        # only show progress bar in one process
        for i, (log, label) in enumerate(tqdm(self.dataset_walker, disable=self.args.local_rank not in [-1, 0])):
            samples.append(deepcopy({"text": log, "label": label}))
        return samples

    def _create_examples(self):
        logger.info("Creating examples")
        self.examples = []
        for sample in tqdm(self.samples, disable=self.args.local_rank not in [-1, 0]):
            label = sample["label"]
            text = sample["text"]

            if self.task_type == "generation":
                if label is None:
                    label = ""
                target = None
                lm_target = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(label))
            else:
                if label is None:
                    label = {"target": False, "linking": None}
                target = label["target"]
                lm_target = None

            text_input = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

            self.examples.append({
                "context": text,
                "text_input": text_input,
                "label": label,
                "target": target,
                "lm_target": lm_target
            })

    def build_input_from_segments(self, context, lm_target=None):
        """ Build a sequence of input from example """
        instance = {"input_ids": self.tokenizer.build_inputs_with_special_tokens(context), "token_type_ids": None}
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        if self.task_type == "generation":
            instance["lm_target_ids"] = self.tokenizer.build_inputs_with_special_tokens(deepcopy(lm_target))
        else:
            instance["lm_target_ids"] = None

        return instance

    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.examples)


class FactLinkingDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(FactLinkingDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = deepcopy(self.examples[index])
        instance = self.build_input_from_segments(example["text_input"])
        instance["label"] = example["target"]
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        mc_token_ids = [ins["mc_token_ids"] for ins in batch]
        labels = [ins["label"] for ins in batch]

        data_info = {}

        pad_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        input_ids = torch.tensor(pad_ids(input_ids, pad_token))
        token_type_ids = torch.full_like(input_ids, 0)
        mc_token_ids = torch.tensor(mc_token_ids)
        lm_labels = torch.full_like(input_ids, 0)
        labels = torch.tensor(labels).long()

        return input_ids, token_type_ids, mc_token_ids, lm_labels, labels, data_info


class FactGenerationDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(FactGenerationDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = deepcopy(self.examples[index])
        instance = self.build_input_from_segments(example["text_input"], example["lm_target"])
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        lm_target_ids = [ins["lm_target_ids"] for ins in batch]

        data_info = {}

        pad_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        input_ids = torch.tensor(pad_ids(input_ids, pad_token))

        token_type_ids = torch.full_like(input_ids, 0)

        decoder_input_ids = torch.tensor(pad_ids(lm_target_ids, pad_token))
        decoder_input_ids = decoder_input_ids[:, :-1].contiguous()

        lm_label_ids = torch.tensor(pad_ids(lm_target_ids, -100))
        lm_label_ids = lm_label_ids[:, 1:].contiguous()

        return input_ids, token_type_ids, decoder_input_ids, lm_label_ids, data_info


class FactGenerationEvalDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(FactGenerationEvalDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = deepcopy(self.examples[index])
        instance = self.build_input_from_segments(example["text_input"], example["lm_target"])
        instance["lm_target_text"] = example["label"]
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        lm_target_text = [ins["lm_target_text"] for ins in batch]

        data_info = {}

        pad_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        input_ids = torch.tensor(pad_ids(input_ids, pad_token))

        token_type_ids = torch.full_like(input_ids, 0)

        return input_ids, token_type_ids, lm_target_text, data_info
