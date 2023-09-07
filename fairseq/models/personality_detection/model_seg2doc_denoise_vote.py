# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)


from fairseq.models.transformer import TransformerEncoder
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules import TransformerDocumentEncoder, LayerNorm
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import safe_getattr, safe_hasattr
from fairseq.modules.fairseq_dropout import FairseqDropout


logger = logging.getLogger(__name__)


@register_model("seg2doc_denoise_vote")
class PDModel(FairseqEncoderModel):
    def __init__(self, args, sen_encoder, trait, vote, **kwargs):
        super().__init__(sen_encoder)
        self.args = args
        self.vote = vote
        self.classification_heads = nn.ModuleDict()
        # We follow BERT's random weight initialization
        logger.info("apply init_bert_params in PDModel")
        self.apply(init_bert_params)

        # twitter
        if trait == 'mbti':
            attn_dim = 111
            self.cls_dim = (1160, 879)
        else:
            attn_dim = 112
            self.cls_dim = (893, 880)

        self.k_proj = nn.Linear(attn_dim, attn_dim, bias=True)
        self.v_proj = nn.Linear(768, 768, bias=True)
        self.q_proj = nn.Linear(attn_dim, attn_dim, bias=True)
        self.out_proj = nn.Linear(768, 128, bias=True)

        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        nn.init.constant_(self.out_proj.bias, 0.0)

        self.scaling = 100 ** -0.5  # scaling can be ignored

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )

        for param in self.parameters():
            param.param_group = "soft"
        for param in self.encoder.parameters():
            param.param_group = "solid"

        initial_layers = self.args.random_initial_layers
        if initial_layers > 0:
            ignore_layers = [self.args.encoder_layers - 1 - i for i in range(initial_layers)]
            for layer in ignore_layers:
                for param in self.encoder.sentence_encoder.layers[layer].parameters():
                    param.param_group = "soft"

        if self.vote:
            self.sigmas_dota = kwargs['sigmas_dota']
            self.register_parameter("sigmas_dota", self.sigmas_dota)

            # self.sigmas_dota.param_group = "param_sigmas"


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--load-checkpoint-heads",
            action="store_true",
            help="(re-)register and load heads when loading checkpoints",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument(
            "--encoder-layerdrop",
            type=float,
            metavar="D",
            default=0,
            help="LayerDrop probability for encoder",
        )
        parser.add_argument(
            "--encoder-layers-to-keep",
            default=None,
            help="which layers to *keep* when pruning as a comma-separated list",
        )
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument(
            "--quant-noise-pq",
            type=float,
            metavar="D",
            default=0,
            help="iterative PQ quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-pq-block-size",
            type=int,
            metavar="D",
            default=8,
            help="block size of quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-scalar",
            type=float,
            metavar="D",
            default=0,
            help="scalar quantization noise and scalar quantization at training time",
        )
        parser.add_argument(
            "--untie-weights-roberta",
            action="store_true",
            help="Untie weights between embeddings and classifiers in RoBERTa",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            default=False,
            help="Apply spectral normalization on the classification head",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        from omegaconf import OmegaConf

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, False)

        # make sure all arguments are present
        base_architecture(args)

        if not safe_hasattr(args, "max_positions"):
            if not safe_hasattr(args, "tokens_per_sample"):
                args.tokens_per_sample = task.max_positions()
            args.max_positions = args.tokens_per_sample

        encoder = RobertaEncoder(args, task.source_dictionary)

        trait = task.cfg.personality

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, True)

        vote = task.cfg.vote
        if vote:
            sigmas_dota = nn.Parameter(torch.tensor([1.0, 0.1]), requires_grad=False)
            return cls(args, encoder, trait, vote, sigmas_dota=sigmas_dota)
        else:
            return cls(args, encoder, trait, vote)

    def forward(
        self,
        src_tokens,  # batch_size_1(documents_num) * batch_size_2(sentence_num) * seq_length
        features_only=False,
        return_all_hiddens=False,
        **kwargs
    ):
        features_only = True

        seg_features = []
        for mairesse, senticnet, emotion in zip(kwargs['seg_mairesse'], kwargs['seg_senticnet'], kwargs['seg_emotion']):
            a = torch.concatenate([mairesse, senticnet, emotion], dim=-1)
            # a = torch.where(torch.isnan(a), torch.full_like(a, 0), a)
            seg_features.append(a)

        doc_features = torch.concatenate([kwargs['doc_mairesse'], kwargs['doc_senticnet'], kwargs['doc_emotion']], dim=-1)
        doc_features = torch.where(torch.isnan(doc_features), torch.full_like(doc_features, 0), doc_features)

        outputs = []
        vote_x = []
        for i, seg in enumerate(src_tokens):  # batch_size_2(seg_num) * seq_length
            x, _ = self.encoder(seg, features_only, return_all_hiddens, **kwargs)
            seg_embeds = x[:, 0, :]   # batch_size_2(seg_num) * 768
            # seg_feats = seg_features[i]
            q = self.q_proj(seg_features[i])
            q *= self.scaling # scaling can be ignored
            k = self.k_proj(seg_features[i])
            v = self.v_proj(seg_embeds)
            attn_weights = torch.matmul(q, k.transpose(0, 1))
            attn_weights_float = utils.softmax(
                attn_weights, dim=-1
            )
            attn_weights = attn_weights_float.type_as(attn_weights)
            attn_probs = self.dropout_module(attn_weights)
            attn = torch.matmul(attn_probs, v)
            doc_rep = self.out_proj(attn)   # 8 * 128
            outputs.append(torch.flatten(doc_rep))  # 1280

            if self.vote:
                v_x = self.classification_heads['segment_classification_head'](seg_embeds[:3, ], seg_features[i][:3, ])
                v_x_sub = self.classification_heads['segment_classification_head_sub'](seg_embeds[3:, ], seg_features[i][3:, ])
                vote_x.append(torch.concatenate((v_x, v_x_sub), dim=0))

        outputs = torch.stack(outputs, dim=0)   # batch_size * 1024
        x = self.classification_heads['document_classification_head'](outputs, doc_features, kwargs['doc_senticnet_dis'])

        return x, (vote_x, self.sigmas_dota)

    def _get_adaptive_head_loss(self):
        norm_loss = 0
        scaling = float(self.args.mha_reg_scale_factor)
        for layer in self.encoder.sentence_encoder.layers:
            norm_loss_layer = 0
            for i in range(layer.self_attn.num_heads):
                start_idx = i * layer.self_attn.head_dim
                end_idx = (i + 1) * layer.self_attn.head_dim
                norm_loss_layer += scaling * (
                    torch.sum(
                        torch.abs(
                            layer.self_attn.q_proj.weight[
                                start_idx:end_idx,
                            ]
                        )
                    )
                    + torch.sum(
                        torch.abs(layer.self_attn.q_proj.bias[start_idx:end_idx])
                    )
                )
                norm_loss_layer += scaling * (
                    torch.sum(
                        torch.abs(
                            layer.self_attn.k_proj.weight[
                                start_idx:end_idx,
                            ]
                        )
                    )
                    + torch.sum(
                        torch.abs(layer.self_attn.k_proj.bias[start_idx:end_idx])
                    )
                )
                norm_loss_layer += scaling * (
                    torch.sum(
                        torch.abs(
                            layer.self_attn.v_proj.weight[
                                start_idx:end_idx,
                            ]
                        )
                    )
                    + torch.sum(
                        torch.abs(layer.self_attn.v_proj.bias[start_idx:end_idx])
                    )
                )

            norm_loss += norm_loss_layer
        return norm_loss

    def _get_adaptive_ffn_loss(self):
        ffn_scale_factor = float(self.args.ffn_reg_scale_factor)
        filter_loss = 0
        for layer in self.encoder.sentence_encoder.layers:
            filter_loss += torch.sum(
                torch.abs(layer.fc1.weight * ffn_scale_factor)
            ) + torch.sum(torch.abs(layer.fc2.weight * ffn_scale_factor))
            filter_loss += torch.sum(
                torch.abs(layer.fc1.bias * ffn_scale_factor)
            ) + torch.sum(torch.abs(layer.fc2.bias * ffn_scale_factor))
        return filter_loss

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log.log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        else:
            if name == 'segment_classification_head':
                self.classification_heads[name] = BasicClassificationHead(
                    input_dim=self.cls_dim[1],
                    inner_dim=64,
                    num_classes=num_classes,
                    activation_fn=self.args.pooler_activation_fn,
                    pooler_dropout=self.args.pooler_dropout,
                    q_noise=self.args.quant_noise_pq,
                    qn_block_size=self.args.quant_noise_pq_block_size,
                    do_spectral_norm=self.args.spectral_norm_classification_head,
                )
                self.classification_heads[name + '_sub'] = BasicClassificationHead(
                    input_dim=self.cls_dim[1],
                    inner_dim=32,
                    num_classes=num_classes,
                    activation_fn=self.args.pooler_activation_fn,
                    pooler_dropout=0.5,
                    q_noise=self.args.quant_noise_pq,
                    qn_block_size=self.args.quant_noise_pq_block_size,
                    do_spectral_norm=self.args.spectral_norm_classification_head,
                )
                nn.init.xavier_uniform_(self.classification_heads[name + '_sub'].dense.weight)
                nn.init.xavier_uniform_(self.classification_heads[name + '_sub'].out_proj.weight)
                if not self.vote:
                    for p_name, param in self.classification_heads[name].named_parameters():
                        logger.info("freeze params in classification_heads." + name + '.' + p_name)
                        param.requires_grad = False

                for param in self.classification_heads[name].parameters():
                    param.param_group = "sub_classifier"
                for param in self.classification_heads[name + '_sub'].parameters():
                    param.param_group = "sub_classifier"
            else:
                self.classification_heads[name] = RobertaClassificationHead(
                    input_dim=self.cls_dim[0],
                    inner_dim=64,
                    num_classes=num_classes,
                    activation_fn=self.args.pooler_activation_fn,
                    pooler_dropout=self.args.pooler_dropout,
                    q_noise=self.args.quant_noise_pq,
                    qn_block_size=self.args.quant_noise_pq_block_size,
                    do_spectral_norm=self.args.spectral_norm_classification_head,
                )
                for param in self.classification_heads[name].parameters():
                    param.param_group = "soft"

    @property
    def supported_targets(self):
        return {"self"}

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""

        # rename decoder -> encoder before upgrading children modules
        for k in list(state_dict.keys()):
            if k.startswith(prefix + "decoder"):
                new_k = prefix + "encoder" + k[len(prefix + "decoder"):]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # rename emb_layer_norm -> layernorm_embedding
        for k in list(state_dict.keys()):
            if ".emb_layer_norm." in k:
                new_k = k.replace(".emb_layer_norm.", ".layernorm_embedding.")
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads."):].split(".")[0]
            # if head_name == 'document_classification_head':
            #     keys_to_delete.append(k)
            #     continue

            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v

        # Copy doc_encoder into the state dict with their current weights.
        if hasattr(self, "doc_encoder"):
            cur_state = self.doc_encoder.state_dict()
            for k, v in cur_state.items():
                if prefix + "doc_encoder." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "doc_encoder." + k)
                    state_dict[prefix + "doc_encoder." + k] = v

        # # Copy vote_param into the state dict with their current weights.
        if hasattr(self, "sigmas_dota") and self.sigmas_dota is not None:
            state_dict[prefix + "sigmas_dota"] = self.sigmas_dota

        # # Copy vote_param into the state dict with their current weights.
        if hasattr(self, "vote_weight") and self.vote_weight is not None:
            state_dict[prefix + "vote_weight"] = self.vote_weight

        # Copy feat_encoder into the state dict with their current weights.
        if hasattr(self, "feat_encoder") and self.feat_encoder is not None:
            cur_state = self.feat_encoder.state_dict()
            for k, v in cur_state.items():
                if prefix + "feat_encoder." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "feat_encoder." + k)
                    state_dict[prefix + "feat_encoder." + k] = v

        for k in list(state_dict.keys()):
            if '.lm_head.' in k:
                del state_dict[k]

        state_dict['k_proj.weight'] = self.k_proj.weight
        state_dict['v_proj.weight'] = self.v_proj.weight
        state_dict['q_proj.weight'] = self.q_proj.weight
        state_dict['out_proj.weight'] = self.out_proj.weight
        state_dict['k_proj.bias'] = self.k_proj.bias
        state_dict['v_proj.bias'] = self.v_proj.bias
        state_dict['q_proj.bias'] = self.q_proj.bias
        state_dict['out_proj.bias'] = self.out_proj.bias

        initial_layers = getattr(self.args, "random_initial_layers", 0)
        if initial_layers:
            cur_state = self.encoder.state_dict()
            ignore_layers = [str((self.args.encoder_layers - 1) - i) for i in range(initial_layers)]
            for k, v in cur_state.items():
                layer = re.search(re.compile(r'layers\.(\d+)\.'), k)
                if layer and layer.group(1) in ignore_layers:
                    k = prefix + "encoder." + k
                    assert k in state_dict.keys()
                    logger.info("initial " + prefix + "layers." + k)
                    state_dict[k] = v


class RobertaEncoder(FairseqEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        # set any missing default values
        base_architecture(args)
        self.args = args

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        embed_tokens = self.build_embedding(
            len(dictionary), args.encoder_embed_dim, dictionary.pad()
        )

        self.sentence_encoder = self.build_encoder(args, dictionary, embed_tokens)

        def freeze_module_params(module, prefix):
            if module is not None:
                for name, param in module.named_parameters():
                    logger.info("freeze RobertaEncoder.sentence_encoder." + prefix + name)
                    param.requires_grad = False

        for layer in range(args.n_trans_layers_to_freeze):
            freeze_module_params(self.sentence_encoder.layers[layer], prefix='layers.' + str(layer) + '.')

        if args.freeze_embeddings:
            freeze_module_params(self.sentence_encoder.embed_tokens, prefix='embed_tokens.')
            freeze_module_params(self.sentence_encoder.embed_positions, prefix='embed_positions.')
            freeze_module_params(self.sentence_encoder.layernorm_embedding, prefix='layernorm_embedding.')

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TransformerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        **unused,
    ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        x, extra = self.extract_features(
            src_tokens, return_all_hiddens=return_all_hiddens
        )
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **kwargs):
        encoder_out = self.sentence_encoder(
            src_tokens,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=kwargs.get("token_embeddings", None),
        )
        # T x B x C -> B x T x C
        features = encoder_out["encoder_out"][0].transpose(0, 1)
        inner_states = encoder_out["encoder_states"] if return_all_hiddens else None
        return features, {"inner_states": inner_states}

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions


class BasicClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        q_noise=0,
        qn_block_size=8,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )
        if do_spectral_norm:
            if q_noise != 0:
                raise NotImplementedError(
                    "Attempting to use Spectral Normalization with Quant Noise. This is not officially supported"
                )
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, seg_reps, seg_features):
        seg_rep = torch.concat((seg_reps, seg_features), dim=-1)
        x = self.dropout(seg_rep)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        num_classes,
        activation_fn,
        pooler_dropout,
        input_dim=512,
        inner_dim=64,
        q_noise=0,
        qn_block_size=8,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )
        self.embed_senticnet_emo = nn.Embedding(25, 5)
        if do_spectral_norm:
            if q_noise != 0:
                raise NotImplementedError(
                    "Attempting to use Spectral Normalization with Quant Noise. This is not officially supported"
                )
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, doc_reps, doc_features, doc_senticnet_dis):
        indicator = self.embed_senticnet_emo(doc_senticnet_dis.int())
        indicator = torch.flatten(indicator, 1)
        x = torch.concatenate((doc_reps, doc_features, indicator), dim=-1)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@register_model_architecture("seg2doc_denoise_vote", "seg2doc_denoise_vote")
def base_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 12)

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = safe_getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = safe_getattr(args, "pooler_dropout", 0.0)
    args.max_source_positions = safe_getattr(args, "max_positions", 512)
    args.no_token_positional_embeddings = safe_getattr(
        args, "no_token_positional_embeddings", False
    )

    # BERT has a few structural differences compared to the original Transformer
    args.encoder_learned_pos = safe_getattr(args, "encoder_learned_pos", True)
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", True)
    args.no_scale_embedding = safe_getattr(args, "no_scale_embedding", True)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = safe_getattr(
        args, "encoder_normalize_before", False
    )
    args.pooler_activation_fn = safe_getattr(args, "pooler_activation_fn", "tanh")
    args.untie_weights_roberta = safe_getattr(args, "untie_weights_roberta", False)

    # Adaptive input config
    args.adaptive_input = safe_getattr(args, "adaptive_input", False)

    # LayerDrop config
    args.encoder_layerdrop = safe_getattr(args, "encoder_layerdrop", 0.0)
    args.encoder_layers_to_keep = safe_getattr(args, "encoder_layers_to_keep", None)

    # Quantization noise config
    args.quant_noise_pq = safe_getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = safe_getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = safe_getattr(args, "quant_noise_scalar", 0)

    # R4F config
    args.spectral_norm_classification_head = safe_getattr(
        args, "spectral_norm_classification_head", False
    )

    # trick for finetune
    args.random_initial_layers = safe_getattr(
        args, "random_initial_layers", 0
    )
    args.freeze_encoder = safe_getattr(
        args, "freeze_encoder", False
    )
    args.n_trans_layers_to_freeze = safe_getattr(
        args, "n_trans_layers_to_freeze", 0
    )

#
# @register_model_architecture("seg2doc", "seg2doc_large")
# def seg_architecture(args):
#     args.encoder_layers = safe_getattr(args, "encoder_layers", 24)
#     args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 1024)
#     args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 4096)
#     args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 16)
#     base_architecture(args)
