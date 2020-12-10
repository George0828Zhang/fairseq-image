# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# import json
# from typing import Any, Dict, List, Optional, Tuple

# import torch
# import torch.nn as nn
# from torch import Tensor
# import pdb
# from fairseq import utils
# from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
)
from fairseq.models.nat import NATransformerModel
from .vggtransformer import VGGEncoder, base_architecture as ar_base_architecture

import logging

logger = logging.getLogger(__name__)


@register_model("vgg_nat")
class VGGNATransformerModel(NATransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        def no_emb_cpy(self, input):
            assert not self.decoder.src_embedding_copy, "do not support embedding copy."
        self.register_forward_pre_hook(no_emb_cpy)

    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)
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

    def initialize_output_tokens(self, encoder_out, src_tokens):
        """ Since src_tokens is float tensor, decoder input tensor
        will be as well. We need to correct tensor type to long. """
        out = super().initialize_output_tokens(encoder_out, src_tokens)
        return out._replace(
            output_tokens=out.output_tokens.long()
        )

@register_model_architecture("vgg_nat", "vgg_nat")
def base_architecture(args):
    ar_base_architecture(args)

    ### nat specific
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)