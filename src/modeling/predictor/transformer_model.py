from transformers import AutoConfig
from modeling_bart import BartModel
from transformers import BartForConditionalGeneration as BartEmbedder
from transformers.modeling_outputs import BaseModelOutput
import torch
import torch as th
import torch.nn as nn
from src.modeling.diffusion.nn import (
    SiLU,
    linear,
    timestep_embedding,
)
import math


def shift_right_2d(input_ids: torch.Tensor, pad_token_id: int, start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.clone().new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class TransformerNetModel_encoder_decoder(nn.Module):
    """
    A transformer model to be used in Diffusion Model Training.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes. TODO for the next version
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            init_pretrained,
            init_pretrained_embedder,
            freeze_embeddings,
            use_pretrained_embeddings,
            dropout=0,
            use_checkpoint=False,
            num_heads=1,
            config=None,
            config_name="facebook/bart-base",
            config_name_embedder="facebook/bart-base",
            vocab_size=None,
            logits_mode=1,
            encoder_layers=6,
            decoder_layers=6,
            embedding_dim=768,
            load_ckpt=None,
            tokenizer=None,
            embedder_args=None
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.dropout = dropout

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.logits_mode = logits_mode
        self.vocab_size = vocab_size
        self.init_pretrained = init_pretrained
        self.init_pretrained_embedder = init_pretrained_embedder
        self.freeze_embeddings = freeze_embeddings
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.config = config
        self.embedder_args = embedder_args
        self.config_name = config_name
        self.config_name_embedder = config_name_embedder
        self.load_ckpt = load_ckpt
        self.tokenizer = tokenizer

        if not self.init_pretrained or not self.init_pretrained_embedder:
            self.config.encoder_layers = encoder_layers
            self.config.decoder_layers = decoder_layers
            self.config.vocab_size = vocab_size
            self.config.encoder_attention_heads = num_heads
            self.config.decoder_attention_heads = num_heads
            self.config.d_model = in_channels
            self.config.encoder_ffn_dim = model_channels
            self.config.decoder_ffn_dim = model_channels
        self.embedding_dim = embedding_dim
        self.embed_scale = math.sqrt(self.embedding_dim) if self.config.scale_embedding else 1.0

        time_embed_dim = in_channels
        self.time_embed = nn.Sequential(
            linear(in_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.d_model),
        )

        self.embedders = None
        self.input_transformers = None
        self.input_up_proj_dec = None
        self.input_up_proj_enc = None
        self.output_down_proj = None
        self.lm_head = None

        self.build_fact_embedder()
        self.build_xstart_predictor()
        self.build_input_output_projections()
        # self.build_embeddings()
        # self.share_encoder_head_embeddings()

        self.LayerNorm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        if self.load_ckpt is not None:
            self.load_weight(self.load_ckpt)

    def get_embeds(self, input_ids):
        return self.input_transformers.decoder.embed_tokens(input_ids) * self.embed_scale

    def load_weight(self, path):
        self.load_state_dict(torch.load(path))
        print(f'weigth initialize from {path}')

    def build_fact_embedder(self):
        temp_bart = BART_Embedder_PT(tokenizer=self.tokenizer, **self.embedder_args)
        # if temp_bart.config.vocab_size != self.vocab_size:
        #     print("Resize Vocabulary: "+str(self.vocab_size))
        #     temp_bart.resize_token_embeddings(self.vocab_size)
        self.embedders = temp_bart
        if "pt" in self.config_name_embedder:
            print("Freeze Pre-Trained Fact Embeddings!")
            for param in self.embedders.parameters():
                param.requires_grad = False

    def build_xstart_predictor(self):
        if self.init_pretrained:
            temp_bart = BartModel.from_pretrained(self.config_name, embedding_dim=self.embedding_dim)
            if temp_bart.config.vocab_size != self.vocab_size:
                print("Resize Vocabulary: "+str(self.vocab_size))
                temp_bart.resize_token_embeddings(self.vocab_size)
            self.input_transformers = temp_bart
        else:
            self.input_transformers = BartModel(self.config, self.embedding_dim)

    def build_input_output_projections(self):
        if not self.init_pretrained and self.in_channels != self.embedding_dim:
            # need to adapt the model to the embedding size
            self.input_up_proj_dec = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, self.config.d_model),
                nn.Tanh(),
                nn.Linear(self.config.d_model, self.config.d_model),
            )

            self.input_up_proj_enc = nn.Sequential(
                nn.Linear(self.embedding_dim, self.config.d_model),
                nn.Tanh(),
                nn.Linear(self.config.d_model, self.config.d_model),
            )

            self.output_down_proj = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_model),
                nn.Tanh(),
                nn.Linear(self.config.d_model, self.embedding_dim),
            )

        else:
            self.input_up_proj_dec = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, self.config.d_model),
                nn.Tanh(),
                nn.Linear(self.config.d_model, self.config.d_model),
            )
            self.input_up_proj_enc = nn.Identity()
            self.output_down_proj = nn.Identity()

    def share_encoder_head_embeddings(self):

        self.embedders.embedder.lm_head.weight = self.embedders.embedder.model.shared.weight

    def build_embeddings(self):

        self.lm_head = nn.Linear(self.embedding_dim, self.input_transformers.shared.weight.shape[0])

        if self.freeze_embeddings:
            with th.no_grad():
                self.lm_head.weight = self.input_transformers.shared.weight
        else:
            self.lm_head.weight = self.input_transformers.shared.weight

    def get_fact_embeds(self, input_ids, attention_mask):
        input_shape = input_ids.shape
        # input_2d = shift_right_2d(input_ids.view(-1, input_shape[-1]),
        #                           self.config.pad_token_id, self.config.bos_token_id)
        input_2d = input_ids.view(-1, input_shape[-1])
        # attention_mask_2d = shift_right_2d(attention_mask.view(-1, input_shape[-1]), 0, 1)
        attention_mask_2d = attention_mask.view(-1, input_shape[-1])
        embeddings_2d = self.embedders.get_fact_embeds(input_2d, attention_mask_2d)
        embeddings = embeddings_2d.view(input_shape[0], input_shape[1], input_shape[-1], -1)
        return embeddings

    def get_fact_logits(self, decoder_input_ids, fact_embeds, labels=None, cross_attention_mask=None):
        input_shape = decoder_input_ids.shape
        fact_embeds_shape = fact_embeds.shape
        input_2d = decoder_input_ids.view(-1, input_shape[-1])
        # input_2d_shifted = shift_right_2d(input_2d, self.config.pad_token_id, self.config.decoder_start_token_id)
        input_2d_shifted = input_2d
        # labels_shifted = input_2d.clone().masked_fill_(input_2d == self.config.pad_token_id, -100)
        labels_shifted = labels.clone().view(-1, input_shape[-1])
        encoder_hidden_states_3d = fact_embeds.view(-1, fact_embeds_shape[-2], fact_embeds_shape[-1])
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states_3d)
        if cross_attention_mask is not None:
            cross_attention_mask_2d = cross_attention_mask.view(-1, fact_embeds_shape[-2])
        else:
            cross_attention_mask_2d = None
        outputs_loss, outputs_logits = self.embedders.forward(labels=labels_shifted,
                                                              decoder_input_ids=input_2d_shifted,
                                                              encoder_outputs=encoder_outputs,
                                                              attention_mask=cross_attention_mask_2d)
        fact_logits = outputs_logits.view(input_shape[0], input_shape[1], input_shape[-1], -1)
        return outputs_loss, fact_logits

    def generate_facts(self, input_fact_embeds, cross_attention_mask=None, de_embed_input_ids=None,
                       gen_args=None, tokenizer=None):
        fact_embeds_shape = input_fact_embeds.shape
        encoder_hidden_states_3d = input_fact_embeds.view(-1, fact_embeds_shape[-2], fact_embeds_shape[-1])
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states_3d)
        if cross_attention_mask is not None:
            cross_attention_mask_2d = cross_attention_mask.view(-1, fact_embeds_shape[-2])
        else:
            cross_attention_mask_2d = None

        gen_ids = self.embedders.generate(encoder_outputs=encoder_outputs,
                                          decoder_input_ids=de_embed_input_ids,
                                          attention_mask=cross_attention_mask_2d,
                                          gen_args=gen_args,
                                          return_ids=True)

        gen_facts = []
        batch_size = fact_embeds_shape[0]
        fact_num = fact_embeds_shape[1]
        for split in range(0, batch_size * fact_num, fact_num):
            gen_facts_batch = tokenizer.batch_decode(gen_ids[split:(split + fact_num), :], skip_special_tokens=False,
                                                     clean_up_tokenization_spaces=True)
            gen_facts_batch = list(map(str.strip, gen_facts_batch))
            '''
            gen_facts_batch = list(map(lambda s: s.split('</s>')[0].strip(), gen_facts_batch))
            gen_facts_batch_clean = []
            for fact in gen_facts_batch:
                gen_facts_batch_clean.append(fact)
                if "<eos_fact>" in fact:
                    break
            '''
            gen_facts.append(gen_facts_batch)

        return gen_facts

    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def forward_encoder(self,
                        input_ids=None,
                        timesteps=None,
                        attention_mask=None,
                        decoder_inputs_embeds=None,
                        decoder_attention_mask=None,
                        self_conditions=None,
                        ):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        emb = self.time_embed(timestep_embedding(timesteps, self.in_channels))
        seq_length = decoder_inputs_embeds.size(1)
        if len(emb.shape) < 3:
            emb = emb.unsqueeze(1).expand(-1, seq_length, -1)
        # decoder_inputs_embeds = self.input_transformers.decoder.embed_tokens(decoder_input_ids) * self.embed_scale
        if self_conditions is not None:
            decoder_inputs_embeds = th.concat((decoder_inputs_embeds, self_conditions), dim=-1)

        decoder_inputs_embeds = (
                self.input_up_proj_dec(decoder_inputs_embeds)
                + emb
        )
        emb_inputs = self.dropout(self.LayerNorm(decoder_inputs_embeds))

        encoder_hidden_states = self.input_transformers(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=self.input_up_proj_enc(
                self.input_transformers.encoder.embed_tokens(input_ids) * self.embed_scale),
            decoder_input_ids=None,
            decoder_inputs_embeds=emb_inputs,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=True,
        ).encoder_last_hidden_state

        return encoder_hidden_states

    def forward(self,
                input_ids=None,
                timesteps=None,
                attention_mask=None,
                decoder_inputs_embeds=None,
                decoder_attention_mask=None,
                self_conditions=None,
                encoder_outputs=None,
                ):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert encoder_outputs is None or input_ids is None
        emb = self.time_embed(timestep_embedding(timesteps, self.in_channels))
        seq_length = decoder_inputs_embeds.size(1)
        if len(emb.shape) < 3:
            emb = emb.unsqueeze(1).expand(-1, seq_length, -1)
        if self_conditions is not None:
            decoder_inputs_embeds = th.concat((decoder_inputs_embeds, self_conditions), dim=-1)

        decoder_inputs_embeds = (
                self.input_up_proj_dec(decoder_inputs_embeds)
                + emb
        )
        emb_inputs = self.dropout(self.LayerNorm(decoder_inputs_embeds))

        input_trans_hidden_states = self.input_transformers(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=self.input_up_proj_enc(self.input_transformers.encoder.embed_tokens(
                input_ids) * self.embed_scale) if input_ids is not None else None,
            decoder_input_ids=None,
            decoder_inputs_embeds=emb_inputs,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs
        ).last_hidden_state

        h = self.output_down_proj(input_trans_hidden_states)

        return h


class BART_Embedder_PT(nn.Module):
    """
    A transformer model to be used in Diffusion Model Training.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes. TODO for the next version
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
            self,
            in_channel,
            num_channels,
            out_channel,
            init_pretrained_embedder,
            freeze_embeddings,
            use_pretrained_embeddings,
            dropout=0,
            use_checkpoint=False,
            num_heads=1,
            config=None,
            config_name_embedder="facebook/bart-base",
            vocab_size=None,
            logits_mode=1,
            encoder_layers=6,
            decoder_layers=6,
            embedding_dim=768,
            load_ckpt=None,
            tokenizer=None,
            **kwargs
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name_embedder)
            config.dropout = dropout

        self.in_channels = in_channel
        self.model_channels = num_channels
        self.out_channels = out_channel
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.logits_mode = logits_mode
        self.vocab_size = vocab_size
        self.init_pretrained_embedder = init_pretrained_embedder
        self.freeze_embeddings = freeze_embeddings
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.config = config
        self.config_name_embedder = config_name_embedder
        self.load_ckpt = load_ckpt

        self.embedding_dim = embedding_dim
        self.embed_scale = math.sqrt(self.embedding_dim) if self.config.scale_embedding else 1.0
        self.tokenizer = tokenizer

        self.embedder = None

        self.build_fact_embedder()
        # self.share_embedder_head_embeddings()

        if self.load_ckpt is not None:
            self.load_weight(self.load_ckpt)

    def build_fact_embedder(self):
        if self.init_pretrained_embedder:
            print("Use Pretrained Embedder: "+self.config_name_embedder)
            temp_bart = BartEmbedder.from_pretrained(self.config_name_embedder)
            if temp_bart.config.vocab_size != self.vocab_size:
                print("Resize Vocabulary: "+str(self.vocab_size))
                temp_bart.resize_token_embeddings(self.vocab_size)
            self.embedder = temp_bart
        else:
            self.embedder = BartEmbedder(self.config)

    def share_embedder_head_embeddings(self):
        self.embedder.lm_head.weight = self.embedder.model.shared.weight

    def get_fact_embeds(self, input_ids, attention_mask):
        # input_shifted = shift_right_2d(input_ids, self.config.pad_token_id, self.config.bos_token_id)
        input_shifted = input_ids
        # attention_mask_shifted = shift_right_2d(attention_mask, 0, 1)
        attention_mask_shifted = attention_mask
        embeddings = self.embedder.get_encoder().forward(input_ids=input_shifted,
                                                         attention_mask=attention_mask_shifted)["last_hidden_state"]
        return embeddings * self.embed_scale

    def load_weight(self, path):
        self.load_state_dict(torch.load(path))
        print(f'weigth initialize from {path}')

    def forward_decoder(self, decoder_input_ids, fact_embeds, labels, cross_attention_mask=None):
        encoder_outputs = BaseModelOutput(last_hidden_state=fact_embeds, hidden_states=None, attentions=None)
        # decoder_input_ids_shifted = shift_right_2d(decoder_input_ids,
        #                                            self.config.pad_token_id, self.config.decoder_start_token_id)
        # labels = decoder_input_ids.clone().masked_fill_(decoder_input_ids == self.config.pad_token_id, -100)
        outputs = self.embedder.forward(labels=labels,
                                        decoder_input_ids=decoder_input_ids,
                                        encoder_outputs=encoder_outputs,
                                        attention_mask=cross_attention_mask)
        return outputs

    def generate_facts(self, input_fact_embeds, cross_attention_mask=None, decoder_input_ids=None, gen_args=None,
                       return_ids=False):
        # fact_embeds_shape = input_fact_embeds.shape
        encoder_outputs = BaseModelOutput(last_hidden_state=input_fact_embeds, hidden_states=None, attentions=None)
        gen_ids = self.embedder.generate(encoder_outputs=encoder_outputs,
                                         attention_mask=cross_attention_mask,
                                         decoder_input_ids=decoder_input_ids,
                                         do_sample=gen_args.fg_do_sample, max_length=gen_args.fg_max_len,
                                         top_k=gen_args.fg_top_k, top_p=gen_args.fg_top_p).cpu().numpy()

        if return_ids:
            return gen_ids
        else:
            return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        # gen_facts = [fact.split('</s>')[0].strip() for fact in gen_facts]

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None,
                encoder_outputs=None, cross_attention_mask=None, dae=False):

        assert encoder_outputs is None or input_ids is None
        if encoder_outputs is None:
            assert input_ids is not None
            embeddings = self.get_fact_embeds(input_ids, attention_mask)
            if dae:
                encoder_outputs = embeddings
                if cross_attention_mask is None:
                    # cross_attention_mask = shift_right_2d(attention_mask, 0, 1)
                    cross_attention_mask = attention_mask
            else:
                encoder_outputs = embeddings[:, 0, :].unsqueeze(1)

        outputs = self.forward_decoder(decoder_input_ids, encoder_outputs, labels, cross_attention_mask)

        return outputs["loss"], outputs["logits"]

    def generate(self, input_ids=None, attention_mask=None, cross_attention_mask=None, encoder_outputs=None,
                 decoder_input_ids=None, gen_args=None, dae=False, return_ids=False, return_embs=False):

        assert encoder_outputs is None or input_ids is None
        if encoder_outputs is None:
            assert input_ids is not None
            embeddings = self.get_fact_embeds(input_ids, attention_mask)
            if dae:
                encoder_outputs = embeddings
                if cross_attention_mask is None:
                    # cross_attention_mask = shift_right_2d(attention_mask, 0, 1)
                    cross_attention_mask = attention_mask
            else:
                encoder_outputs = embeddings[:, 0, :].unsqueeze(1)

        facts = self.generate_facts(encoder_outputs, cross_attention_mask, decoder_input_ids, gen_args, return_ids)

        if return_embs and not dae:
            fact_embs = encoder_outputs.squeeze(1).detach().cpu().numpy().tolist()
            return facts, fact_embs
        else:
            return facts
