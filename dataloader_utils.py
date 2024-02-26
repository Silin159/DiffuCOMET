import logging
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from functools import partial
from mpi4py import MPI
import json
import numpy as np
from copy import deepcopy

logging.basicConfig(level=logging.INFO)


def get_dataloader(tokenizer, data_path, batch_size, max_seq_len, max_seq_len_src, args):
    dataset = TextDatasetSeq2Seq(tokenizer=tokenizer, data_path=data_path, source=args.src, target=args.tgt,
                                 shard=MPI.COMM_WORLD.Get_rank(),
                                 num_shards=MPI.COMM_WORLD.Get_size())

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # 20,
        drop_last=True,
        shuffle='train' in data_path,
        num_workers=10,
        collate_fn=partial(TextDatasetSeq2Seq.collate_pad,
                           args=args,
                           cutoff=max_seq_len,
                           cutoff_src=max_seq_len_src,
                           padding_token=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else
                           tokenizer.get_vocab()['<pad>'])
    )

    while True:
        for batch in dataloader:
            yield batch


def get_dataloader_kg(tokenizer, data_path, batch_size,  max_seq_len_src, max_seq_len, max_fact_len, args):
    dataset = TextDatasetSeq2KG(tokenizer=tokenizer, data_path=data_path, source=args.src, target=args.tgt,
                                shard=MPI.COMM_WORLD.Get_rank(),
                                num_shards=MPI.COMM_WORLD.Get_size())

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # 20,
        drop_last=True,
        shuffle='train' in data_path,
        num_workers=10,
        collate_fn=partial(TextDatasetSeq2KG.collate_pad,
                           args=args,
                           cutoff_src=max_seq_len_src,
                           cutoff_tgt=max_seq_len,
                           cutoff_fact=max_fact_len,
                           padding_token=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else
                           tokenizer.get_vocab()['<pad>'],
                           bos_token=tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else
                           tokenizer.get_vocab()['<s>'],
                           eos_token=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else
                           tokenizer.get_vocab()['</s>'],
                           decoder_start_token=tokenizer.decoder_start_token_id if
                           hasattr(tokenizer, 'decoder_start_token_id') else tokenizer.get_vocab()['</s>'],
                           eos_fact=tokenizer.get_vocab()['<eos_fact>'])
        )

    while True:
        for batch in dataloader:
            yield batch


def get_dataloader_pte(tokenizer, data_path, batch_size, max_fact_len, args):
    dataset = TextDatasetPTE(tokenizer=tokenizer, data_path=data_path,
                             shard=MPI.COMM_WORLD.Get_rank(),
                             num_shards=MPI.COMM_WORLD.Get_size())

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle='train' in data_path,
        num_workers=10,
        collate_fn=partial(TextDatasetPTE.collate_pad,
                           args=args,
                           cutoff_fact=max_fact_len,
                           padding_token=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else
                           tokenizer.get_vocab()['<pad>'],
                           bos_token=tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else
                           tokenizer.get_vocab()['<s>'],
                           eos_token=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else
                           tokenizer.get_vocab()['</s>'],
                           decoder_start_token=tokenizer.decoder_start_token_id if
                           hasattr(tokenizer, 'decoder_start_token_id') else tokenizer.get_vocab()['</s>'])
    )

    while True:
        for batch in dataloader:
            yield batch


class TextDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            data_path: str,
            has_labels: bool = False
    ) -> None:

        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.read_data()
        if has_labels:
            self.read_labels()

    def read_data(self):
        logging.info("Reading data from {}".format(self.data_path))
        data = pd.read_csv(self.data_path, sep="\t", header=None)  # read text file
        logging.info(f"Tokenizing {len(data)} sentences")

        self.text = data[0].apply(lambda x: x.strip()).tolist()
        if hasattr(self.tokenizer, 'encode_batch'):

            encoded_input = self.tokenizer.encode_batch(self.text)
            self.input_ids = [x.ids for x in encoded_input]

        else:
            encoded_input = self.tokenizer(self.text)
            self.input_ids = encoded_input["input_ids"]

    def read_labels(self):
        self.labels = pd.read_csv(self.data_path, sep="\t", header=None)[1].tolist()
        # check if labels are already numerical
        self.labels = [str(x) for x in self.labels]
        if isinstance(self.labels[0], int):
            return
        # if not, convert to numerical
        all_labels = sorted(list(set(self.labels)))
        self.label_to_idx = {label: i for i, label in enumerate(all_labels)}
        self.idx_to_label = {i: label for i, label in self.label_to_idx.items()}
        self.labels = [self.label_to_idx[label] for label in self.labels]

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, i):
        out_dict = {
            "input_ids": self.input_ids[i],
            # "attention_mask": [1] * len(self.input_ids[i]),
        }
        if hasattr(self, "labels"):
            out_dict["label"] = self.labels[i]
        return out_dict

    @staticmethod
    def collate_pad(batch, cutoff: int):
        max_token_len = 0
        num_elems = len(batch)
        # batch[0] -> __getitem__[0] --> returns a tuple (embeddings, out_dict)

        for i in range(num_elems):
            max_token_len = max(max_token_len, len(batch[i]["input_ids"]))

        max_token_len = min(cutoff, max_token_len)

        tokens = torch.zeros(num_elems, max_token_len).long()
        tokens_mask = torch.zeros(num_elems, max_token_len).long()

        has_labels = False
        if "label" in batch[0]:
            labels = torch.zeros(num_elems).long()
            has_labels = True

        for i in range(num_elems):
            toks = batch[i]["input_ids"]
            length = len(toks)
            tokens[i, :length] = torch.LongTensor(toks)
            tokens_mask[i, :length] = 1
            if has_labels:
                labels[i] = batch[i]["label"]

        # TODO: the first return None is just for backward compatibility -- can be removed
        if has_labels:
            return None, {"input_ids": tokens, "attention_mask": tokens_mask, "labels": labels}
        else:
            return None, {"input_ids": tokens, "attention_mask": tokens_mask}


class TextDatasetSeq2Seq(Dataset):

    def __init__(
            self,
            tokenizer,
            data_path: str,
            source,
            target,
            shard,
            num_shards,
    ) -> None:

        super().__init__()
        self.src_text = None
        self.tgt_text = None
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.shard = shard
        self.src = source
        self.tgt = target
        self.num_shards = num_shards
        self.read_data()

    def read_data(self):
        print("Reading data from {}".format(self.data_path))
        data = [open(self.data_path + '.' + self.src, 'r').readlines(),
                open(self.data_path + '.' + self.tgt, 'r').readlines()]
        print(f"Tokenizing {len(data[0])} sentences")

        data = [[src, tgt] for src, tgt in zip(data[0], data[1])]
        # random.shuffle(data)

        self.src_text = [item[0].strip('\n') for item in data]
        self.tgt_text = [item[1].strip('\n') for item in data]

        bos_idx = (len(self.src_text) // self.num_shards) * self.shard
        eos_idx = (len(self.src_text) // self.num_shards) * (self.shard + 1)
        self.src_text = self.src_text[bos_idx:eos_idx]
        self.tgt_text = self.tgt_text[bos_idx:eos_idx]

        print('examples src', self.src_text[0])
        print('examples tgt', self.tgt_text[0])

        # check if tokenizer has a method 'encode_batch'
        if hasattr(self.tokenizer, 'encode_batch'):

            encoded_input_src = self.tokenizer.encode_batch(self.src_text)
            self.input_ids_src = [x.ids for x in encoded_input_src]

            encoded_input_tgt = self.tokenizer.encode_batch(self.tgt_text)
            self.input_ids_tgt = [x.ids for x in encoded_input_tgt]

        else:

            encoded_input_src = self.tokenizer(self.src_text)
            self.input_ids_src = encoded_input_src["input_ids"]

            encoded_input_tgt = self.tokenizer(self.tgt_text)
            self.input_ids_tgt = encoded_input_tgt["input_ids"]

        count_length_src = np.mean([len(item) for item in self.input_ids_src])
        count_length_tgt = np.mean([len(item) for item in self.input_ids_tgt])

        print(f'average number of tokens in source {count_length_src}')
        print(f'average number of tokens in target {count_length_tgt}')

    def __len__(self) -> int:
        return len(self.src_text)

    def __getitem__(self, i):
        out_dict = {
            "encoder_input_ids": self.input_ids_src[i],
            "decoder_input_ids": self.input_ids_tgt[i],
        }
        return out_dict

    @staticmethod
    def collate_pad(batch, args, cutoff: int, cutoff_src: int, padding_token: int):
        max_token_len_src, max_token_len_tgt = cutoff_src, cutoff
        num_elems = len(batch)

        tokens_src = torch.ones(num_elems, max_token_len_src).long() * padding_token
        tokens_mask_src = torch.zeros(num_elems, max_token_len_src).long()

        tokens_tgt = torch.ones(num_elems, max_token_len_tgt).long() * padding_token
        tokens_mask_tgt = torch.zeros(num_elems, max_token_len_tgt).long()

        for i in range(num_elems):
            toks_src = batch[i]["encoder_input_ids"][:max_token_len_src]
            toks_tgt = batch[i]["decoder_input_ids"][:max_token_len_tgt]
            l_s, l_t = len(toks_src), len(toks_tgt)
            tokens_src[i, :l_s] = torch.LongTensor(toks_src)
            tokens_tgt[i, :l_t] = torch.LongTensor(toks_tgt)
            tokens_mask_src[i, :l_s] = 1
            tokens_mask_tgt[i, :] = 1

        return {"input_ids": tokens_src, "attention_mask": tokens_mask_src,
                'decoder_input_ids': tokens_tgt, 'decoder_attention_mask': tokens_mask_tgt}, None


class TextDatasetSeq2KG(Dataset):

    def __init__(
            self,
            tokenizer,
            data_path: str,
            source,
            target,
            shard,
            num_shards,
    ) -> None:

        super().__init__()
        self.src_text = None
        self.tgt_text = None
        self.src_text_tokens = None
        self.tgt_text_tokens = None
        self.input_ids_src = None
        self.input_ids_tgt = None
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.shard = shard
        self.src = source
        self.tgt = target
        self.num_shards = num_shards
        self.read_data()

    def read_data(self):
        print("Reading data from {}".format(self.data_path))
        source_data = open(self.data_path + '.' + self.src, 'r').readlines()
        print(f"Tokenizing {len(source_data)} sentences")

        self.src_text = [item.strip('\n') for item in source_data]
        self.tgt_text = json.load(open(self.data_path + '.' + self.tgt + '.json', 'r'))

        bos_idx = (len(self.src_text) // self.num_shards) * self.shard
        eos_idx = (len(self.src_text) // self.num_shards) * (self.shard + 1)
        self.src_text = self.src_text[bos_idx:eos_idx]
        self.tgt_text = self.tgt_text[bos_idx:eos_idx]

        print('examples src', self.src_text[0])
        print('examples tgt', self.tgt_text[0])

        self.src_text_tokens = [self.tokenizer.tokenize(item) for item in self.src_text]
        self.tgt_text_tokens = [[self.tokenizer.tokenize(item) for item in facts] for facts in self.tgt_text]

        self.input_ids_src = [self.tokenizer.convert_tokens_to_ids(item) for item in self.src_text_tokens]
        self.input_ids_tgt = [[self.tokenizer.convert_tokens_to_ids(item) for item in facts]
                              for facts in self.tgt_text_tokens]

        count_length_src = np.mean([len(item) for item in self.input_ids_src])
        count_length_tgt_facts = np.mean([len(item) for item in self.input_ids_tgt])
        # count_length_tgt_tokens = np.mean([[len(item) for item in facts] for facts in self.input_ids_tgt])

        print(f'average number of tokens in source {count_length_src}')
        print(f'average number of facts in target {count_length_tgt_facts}')
        # print(f'average number of fact tokens in target {count_length_tgt_tokens}')

    def __len__(self) -> int:
        return len(self.src_text)

    def __getitem__(self, i):
        out_dict = {
            "encoder_input_ids": self.input_ids_src[i],
            "decoder_input_ids": self.input_ids_tgt[i],
        }
        return out_dict

    @staticmethod
    def collate_pad(batch, args, cutoff_src: int, cutoff_tgt: int, cutoff_fact: int,
                    padding_token: int, bos_token: int, eos_token: int, decoder_start_token: int,
                    eos_fact: int):

        max_token_len_src, max_fact_len_tgt, max_fact_token_len_tgt = cutoff_src, cutoff_tgt, cutoff_fact
        num_elems = len(batch)

        tokens_src = torch.ones(num_elems, max_token_len_src).long() * padding_token
        tokens_mask_src = torch.zeros(num_elems, max_token_len_src).long()

        tokens_enc_tgt = torch.ones(num_elems, max_fact_len_tgt, max_fact_token_len_tgt).long() * padding_token
        tokens_tgt = torch.ones(num_elems, max_fact_len_tgt, max_fact_token_len_tgt).long() * padding_token
        labels = torch.ones(num_elems, max_fact_len_tgt, max_fact_token_len_tgt).long() * padding_token
        tokens_mask_tgt = torch.zeros(num_elems, max_fact_len_tgt).long()
        tokens_mask_tgt_fact = torch.zeros(num_elems, max_fact_len_tgt).long()
        tokens_mask_tgt_fact_text = torch.zeros(num_elems, max_fact_len_tgt, max_fact_token_len_tgt).long()

        for i in range(num_elems):
            toks_src = batch[i]["encoder_input_ids"][:max_token_len_src]
            l_s = len(toks_src)
            tokens_src[i, :l_s] = torch.LongTensor(toks_src)
            tokens_mask_src[i, :l_s] = 1
            tokens_mask_tgt[i, :] = 1
            for j in range(max_fact_len_tgt):
                if j < len(batch[i]["decoder_input_ids"][:max_fact_len_tgt-1]):
                    toks_enc_tgt = [bos_token] + batch[i]["decoder_input_ids"][j][:max_fact_token_len_tgt-3] + [eos_token]
                elif j == len(batch[i]["decoder_input_ids"][:max_fact_len_tgt-1]):
                    toks_enc_tgt = [bos_token, eos_fact, eos_token]
                else:
                    toks_enc_tgt = None

                if toks_enc_tgt is not None:
                    l_f = len(toks_enc_tgt)
                    toks_tgt = [decoder_start_token] + deepcopy(toks_enc_tgt)
                    toks_labels = deepcopy(toks_enc_tgt)
                    tokens_enc_tgt[i, j, :l_f] = torch.LongTensor(toks_enc_tgt)
                    tokens_tgt[i, j, :l_f+1] = torch.LongTensor(toks_tgt)
                    labels[i, j, :l_f] = torch.LongTensor(toks_labels)
                    tokens_mask_tgt_fact[i, j] = 1
                    tokens_mask_tgt_fact_text[i, j, :l_f] = 1

            labels.masked_fill_(labels == padding_token, -100)
            # if training with dae, randomly mask target tokens for reconstruction, else not masked
            # to do

        return {"input_ids": tokens_src, "attention_mask": tokens_mask_src,
                'embedder_input_ids': tokens_tgt, 'decoder_attention_mask': tokens_mask_tgt,
                'embedder_enc_input_ids': tokens_enc_tgt, "embedder_labels": labels,
                'embedder_fact_mask': tokens_mask_tgt_fact,
                'embedder_attention_mask': tokens_mask_tgt_fact_text}, None


class TextDatasetPTE(Dataset):

    def __init__(
            self,
            tokenizer,
            data_path: str,
            shard,
            num_shards,
    ) -> None:

        super().__init__()
        self.fact_text_3d = None
        self.fact_text = None
        self.fact_text_tokens = None
        self.input_fact_ids = None
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.shard = shard
        self.num_shards = num_shards
        self.read_data()

    def read_data(self):
        print("Reading data from {}".format(self.data_path))

        self.fact_text = json.load(open(self.data_path, 'r'))

        bos_idx = (len(self.fact_text) // self.num_shards) * self.shard
        eos_idx = (len(self.fact_text) // self.num_shards) * (self.shard + 1)
        self.fact_text = self.fact_text[bos_idx:eos_idx]

        self.fact_text_tokens = [self.tokenizer.tokenize(item) for item in self.fact_text]

        self.input_fact_ids = [self.tokenizer.convert_tokens_to_ids(item) for item in self.fact_text_tokens]

        # count_length_fact = np.mean([len(item) for item in self.input_fact_ids])
        # print(f'average number of tokens in facts {count_length_fact }')

    def __len__(self) -> int:
        return len(self.fact_text)

    def __getitem__(self, i):
        out_dict = {
            "input_fact_ids": self.input_fact_ids[i]
        }
        return out_dict

    @staticmethod
    def collate_pad(batch, args, cutoff_fact: int, padding_token: int, bos_token: int,
                    eos_token: int, decoder_start_token: int):

        max_fact_token_len = cutoff_fact
        num_elems = len(batch)

        tokens = torch.ones(num_elems, max_fact_token_len).long() * padding_token
        tokens_mask = torch.zeros(num_elems, max_fact_token_len).long()
        tokens_dec = torch.ones(num_elems, max_fact_token_len).long() * padding_token
        labels = torch.ones(num_elems, max_fact_token_len).long() * padding_token

        for i in range(num_elems):
            toks = [bos_token] + batch[i]["input_fact_ids"][:max_fact_token_len-3] + [eos_token]
            toks_dec = [decoder_start_token] + deepcopy(toks)
            toks_label = deepcopy(toks)
            l_f = len(toks)
            tokens[i, :l_f] = torch.LongTensor(toks)
            tokens_mask[i, :l_f] = 1
            tokens_dec[i, :l_f+1] = torch.LongTensor(toks_dec)
            labels[i, :l_f] = torch.LongTensor(toks_label)

        labels.masked_fill_(labels == padding_token, -100)
        # if training with dae, randomly mask target tokens for reconstruction, else not masked
        # to do

        return {"input_ids": tokens, "attention_mask": tokens_mask,
                "decoder_input_ids": tokens_dec, "labels": labels}, None
