# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import pdb

from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
)
from fairseq.models.nat.insertion_transformer import (
    neg_scorer,
    InsertionTransformerModel
)
from .vggtransformer import VGGEncoder, base_architecture as ar_base_architecture

import logging
logger = logging.getLogger(__name__)


def _get_ins_targets(in_tokens, out_tokens, padding_idx, unk_idx, vocab_size, tau=None):
    """Fix dtype error when tau != None. otherwise same as original.
    """

    try:
        from fairseq import libnat
    except ImportError as e:
        import sys

        sys.stderr.write("ERROR: missing libnat. run `pip install --editable .`\n")
        raise e

    B = in_tokens.size(0)
    T = in_tokens.size(1)
    V = vocab_size

    with torch.cuda.device_of(in_tokens):
        in_tokens_list = [
            [t for t in s if t != padding_idx] for i, s in enumerate(in_tokens.tolist())
        ]
        out_tokens_list = [
            [t for t in s if t != padding_idx]
            for i, s in enumerate(out_tokens.tolist())
        ]

    full_labels = libnat.suggested_ed2_path(
        in_tokens_list, out_tokens_list, padding_idx
    )
    insert_labels = [a[:-1] for a in full_labels]

    # numericalize1
    insert_label_tensors = in_tokens.new_zeros(B * (T - 1) * V).float()
    insert_index, insert_labels = zip(
        *[
            (w + (j + i * (T - 1)) * V, neg_scorer(k, len(label), tau))
            for i, labels in enumerate(insert_labels)
            for j, label in enumerate(labels[1:-1])
            for k, w in enumerate(label)
        ]
    )  # HACK 1:-1
    insert_index, insert_labels = [
        torch.tensor(list(a), device=in_tokens.device)
        for a in [insert_index, insert_labels]
    ]
    insert_label_tensors.scatter_(0, insert_index.long(), insert_labels.type_as(insert_label_tensors))
    insert_label_tensors = insert_label_tensors.view(B, T - 1, V)

    return insert_label_tensors


@register_model("vgg_insertion_transformer")
class VGGInsertionTransformerModel(InsertionTransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @staticmethod
    def add_args(parser):
        InsertionTransformerModel.add_args(parser)
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

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):

        assert tgt_tokens is not None, "forward function only supports training."

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # generate training labels for insertion
        word_ins_out = self.decoder.forward_word_ins(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
        )

        word_ins_tgt = _get_ins_targets(
            prev_output_tokens,
            tgt_tokens,
            self.pad,
            self.unk,
            len(self.tgt_dict),
            tau=self.decoder.label_tau,
        ).type_as(word_ins_out)
        word_ins_masks = prev_output_tokens[:, 1:].ne(self.pad)

        return {
            "word_ins": {
                "out": word_ins_out,
                "tgt": word_ins_tgt,
                "mask": word_ins_masks,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            }
        }

@register_model_architecture("vgg_insertion_transformer", "vgg_insertion_transformer")
def base_architecture(args):
    ar_base_architecture(args)

    # special for insertion transformer
    args.apply_bert_init = getattr(args, "apply_bert_init", False)
    args.label_tau = getattr(args, "label_tau", None)