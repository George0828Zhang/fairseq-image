# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
from fairseq.data import FairseqDataset

# from . import data_utils
from .collaters import Seq2SeqCollater

from PIL import Image
from torchvision import transforms
import imagesize

class ImageDataset(FairseqDataset):
    """
    A dataset representing speech and corresponding transcription.

    Args:
        img_paths: (List[str]): A list of str with paths to image files.
        tgt (List[torch.LongTensor]): A list of LongTensors containing the indices
            of target transcriptions.
        tgt_dict (~fairseq.data.Dictionary): target vocabulary.
        ids (List[str]): A list of utterance IDs.
    """

    def __init__(
        self,
        img_paths,
        tgt,
        tgt_dict,
        ids,
    ):
        assert len(img_paths) > 0
        assert len(img_paths) == len(tgt)
        assert len(img_paths) == len(ids)
        self.img_paths = img_paths
        self.tgt_dict = tgt_dict
        self.tgt = tgt
        self.ids = ids

        self.s2s_collater = Seq2SeqCollater(
            feature_index=0,
            label_index=1,
            pad_index=self.tgt_dict.pad(),
            eos_index=self.tgt_dict.eos(),
            move_eos_to_beginning=True,
        )

        # self.images = []
        self.frame_sizes = []
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.,), (1.,)),
        ])
        for path in self.img_paths:
            if not os.path.exists(path):
                raise FileNotFoundError("Image file not found: {}".format(path))
            # img = Image.open(path)
            # img_tensor = transform(img)
            # self.images.append(img_tensor.detach())
            _w, _h = imagesize.get(path)
            self.frame_sizes.append(  # (C, H, W)
                _w # use W as frame size
            )

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None

        path = self.img_paths[index]
        img = Image.open(path)
        img_tensor = self.transform(img)
        return {"id": index, "data": [img_tensor, tgt_item]}

    def __len__(self):
        return len(self.img_paths)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[int]): sample indices to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        return self.s2s_collater.collate(samples)

    def num_tokens(self, index):
        return self.frame_sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.frame_sizes[index],
            len(self.tgt[index]) if self.tgt is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self))
