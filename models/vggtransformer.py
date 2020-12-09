# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import pdb
from fairseq import utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerModel,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    FairseqDropout,
    PositionalEmbedding,
    # VGGBlock
)

logger = logging.getLogger(__name__)

@register_model("vgg_at")
class VGGAutoregressiveTransformerModel(TransformerModel):
    """
    {
#     "in_channels": 64,
#     "mid_channels": 64,
#     "subsample": 2,
#     "out_channels": 64,
#     "activation_fn": "gelu"
# }
    """

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        """Add task-specific arguments to the parser."""

        parser.add_argument(
            "--vgg-config",
            type=str,
            help="""config in json format e.g. '[{"in_channels":64, "subsample": 2}, {"in_channels":64, "subsample": 2}]'.
             If a dict is empty, default values are used.""",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        tgt_dict = task.target_dictionary
        
        decoder_embed_tokens = cls.build_embedding(
            args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
        )

        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args):
        return VGGEncoder(args)


class VGGNet(nn.Module):
    def __init__(self, in_channels=3, mid_channels=64, out_channels=64, 
    conv_kernel_size=3, subsample=2, 
    num_conv_layers=2, padding=None, activation_fn="relu"):
        super().__init__()
        self.conv_kernel_size = conv_kernel_size
        self.subsample = subsample
        self.layers = nn.ModuleList()
        self.padding = conv_kernel_size//2 if padding is None else padding
        # self.pooling_kernel_size = pooling_kernel_size
        self.activation_fn = utils.get_activation_fn(
            activation=activation_fn
        )

        for layer in range(num_conv_layers):
            conv_op = nn.Conv2d(
                in_channels if layer == 0 else mid_channels,
                out_channels if layer == num_conv_layers-1 else mid_channels,
                self.conv_kernel_size,
                stride=self.subsample if layer == 0 else 1, # subsample at start
                padding=self.padding,
            )
            self.layers.append(conv_op)
            # self.layers.append(self.activation_fn)

        # if self.pooling_kernel_size is not None:
        #     pool_op = nn.MaxPool2d(kernel_size=self.pooling_kernel_size, ceil_mode=True)
        #     self.layers.append(pool_op)
    
    def forward(self, x, return_all_hiddens=False):
        encoder_states = []
        # encoder layers
        for layer in self.layers:
            x = layer(x)
            x = self.activation_fn(x)
            if return_all_hiddens:
                encoder_states.append(x)
        return x, encoder_states




class VGGEncoder(FairseqEncoder):
    def __init__(self, args):
        super().__init__(None) # no dictionary
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.max_source_positions = args.max_source_positions
        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                args.encoder_embed_dim,
                padding_idx=0,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
        if args.encoder_layers < 2:
            raise ValueError(
                "encoder_layers should be larger than 2, got: {}".format(
                    args.encoder_layers
                )
            )
        vgg_config = json.loads(getattr(args, "vgg_config", "{}") or "{}")
        logger.info(vgg_config)
        self.vggblocks = nn.ModuleList([
            VGGNet(**vgg_config[0]
                # in_channels=3, 
                # subsample=3, 
                # conv_kernel_size=5,
                # activation_fn="gelu"
            )
        ])
        for i in range(1,args.encoder_layers):

            self.vggblocks.append(
                VGGNet(
                    **vgg_config[i]
                    # in_channels=64,
                    # subsample=2,
                    # out_channels=args.encoder_embed_dim if i==args.encoder_layers-1 else 64,
                    # activation_fn="gelu"
                )
            )            

    def forward(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):        
        # B x C x H x W
        x = src_tokens
        # TODO: compute padding mask from lengths
        # see causal ST encoder (w/ subsampler) for reference of how to get mask
        # from fairseq.data.data_utils import lengths_to_padding_mask

        encoder_padding_mask = None   
        encoder_states = [] if return_all_hiddens else None
        
        # encoder layers
        for block in self.vggblocks:
            x, extra1 = block(x, return_all_hiddens=return_all_hiddens)
            if return_all_hiddens:
                encoder_states.extend(extra1)

        # B x C x H x W -> B x (HxW) x C
        _b, _c, _h, _w = x.size()
        x = x.view(_b, _c, _h*_w).permute(0, 2, 1)


        if self.embed_positions is not None:
            pos = x.new_ones((_b, _h*_w)) # B x T
            if pos.size(-1) > self.max_positions():                
                pdb.set_trace()
                raise ValueError(
                    "tokens exceeds maximum length: {} > {}".format(
                        pos.size(-1), self.max_positions()
                    )
                )
            x = x + self.embed_positions(pos) # input: B x T

        # B x (HxW) x C -> (HxW) x B x C 
        x = x.permute(1, 0, 2)
        x = self.dropout_module(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=None,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)


@register_model_architecture("vgg_at", "vgg_at")
def base_architecture(args):
    # args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    # args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    # args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    # args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.decoder_embed_dim*4
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)

    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    ### conv
    args.vgg_config = getattr(args, "vgg_config",
        json.dumps([
                {
                    "in_channels": 1,
                    "mid_channels": 64,
                    "out_channels": 128,
                    "conv_kernel_size": 5,
                    "subsample": 3,
                    "activation_fn": "gelu"
                },
                {
                    "in_channels": 128,
                    "mid_channels": 256,
                    "out_channels": 256,
                    "activation_fn": "gelu"
                },
                {
                    "in_channels": 256,
                    "mid_channels": 512,
                    "out_channels": 512,
                    "activation_fn": "gelu"
                },
        ])
    ) 
    