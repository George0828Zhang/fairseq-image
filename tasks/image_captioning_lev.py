# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re
import sys
import logging
import pdb
import itertools
import numpy as np
from argparse import Namespace

import torch
from fairseq import metrics, options, utils
from fairseq.data import (
    Dictionary,
    data_utils,
    encoders,
)
from fairseq.tasks import LegacyFairseqTask, register_task

from .image_captioning import ImageCaptioningTask
# from .data import ImageDataset

# EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)

@register_task('image_captioning_lev')
class ImageCaptioningLevenshteinTask(ImageCaptioningTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        ImageCaptioningTask.add_args(parser)
        parser.add_argument(
            '--noise',
            default='no_noise',
            choices=['random_mask', 'no_noise', 'full_mask'])

    # inherit from translationtask
    # def load_dataset(self, split, epoch=1, combine=False, **kwargs):

    def inject_noise(self, target_tokens):
        pad = self.tgt_dict.pad()
        bos = self.tgt_dict.bos()
        eos = self.tgt_dict.eos()
        unk = self.tgt_dict.unk()

        def _random_mask(target_tokens):
            target_masks = target_tokens.ne(pad) & \
                target_tokens.ne(bos) & \
                target_tokens.ne(eos)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            # make sure to mask at least one token.
            target_length = target_length + 1

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(
                target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk)
            return prev_target_tokens

        def _full_mask(target_tokens):
            target_mask = target_tokens.eq(bos) | target_tokens.eq(
                eos) | target_tokens.eq(pad)
            return target_tokens.masked_fill(~target_mask, unk)

        if self.args.noise == 'random_mask':
            return _random_mask(target_tokens)
        elif self.args.noise == 'full_mask':
            return _full_mask(target_tokens)
        elif self.args.noise == 'no_noise':
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, models, args):
        # add models input to match the API for SequenceGenerator
        from fairseq.iterative_refinement_generator import IterativeRefinementGenerator
        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, 'iter_decode_eos_penalty', 0.0),
            max_iter=getattr(args, 'iter_decode_max_iter', 10),
            beam_size=getattr(args, 'iter_decode_with_beam', 1),
            reranking=getattr(
                args, 'iter_decode_with_external_reranker', False),
            decoding_format=getattr(args, 'decoding_format', None),
            adaptive=not getattr(args, 'iter_decode_force_max_iter', False),
            retain_history=getattr(args, 'retain_iter_history', False))

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        """ *Removes append_bos* """
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError(
                "Constrained decoding with the translation_lev task is not supported")
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary,
                                   tgt_dict=self.target_dictionary)

    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   update_num,
                   ignore_grad=False):
        model.train()
        sample['prev_target'] = self.inject_noise(sample['target'])
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        sample['prev_target'] = self.inject_noise(sample['target'])
        return super().valid_step(sample, model, criterion)