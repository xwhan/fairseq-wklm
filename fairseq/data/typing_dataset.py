# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from . import data_utils, FairseqDataset


def collate(samples, pad_idx, num_class):

    if len(samples) == 0:
        return {}
    if not isinstance(samples[0], dict):
        samples = [s for sample in samples for s in sample]

    batch_text = data_utils.collate_tokens(
        [s['text'] for s in samples], pad_idx, left_pad=False)


    masks = torch.zeros(batch_text.size(0), batch_text.size(1))
    target = torch.zeros(batch_text.size(0), num_class)

    for idx, s in enumerate(samples):
        e_offset = s['e_offset']
        masks[idx, e_offset] = 1
        for t in s['target']:
            target[idx, t] = 1

    return {
        'ntokens': sum(len(s['text']) for s in samples),
        'net_input': {
            'sentence': batch_text,
            'segment': data_utils.collate_tokens(
                [s['segment'] for s in samples], pad_idx, left_pad=False,
            ),
            'entity_masks': masks
        },
        'target': target,
        'nsentences': len(samples),
    }


class TypingDataset(FairseqDataset):
    """
    Args:
        dataset (torch.utils.data.Dataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
    """

    def __init__(self, dataset, e_offsets, e_lens, loaded_labels, sizes, dictionary, max_length, num_class, shuffle=False, use_marker=False, use_sep=False):
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.e_offsets = e_offsets
        self.e_lens = e_lens

        self.vocab = dictionary
        self.shuffle = shuffle
        self.max_length = max_length
        self.use_marker = use_marker
        self.use_sep = use_sep
        self.labels = loaded_labels
        self.num_class = num_class

    def __getitem__(self, index):
        block_text = self.dataset[index]
        e_offset_orig = self.e_offsets[index]
        e_len = self.e_lens[index]
        e_end = e_offset_orig + e_len
        type_label = self.labels[index]
        block_text = block_text.tolist()

        if self.use_marker:
            e_start_marker = self.vocab.index("[unused1]")
            e_end_marker = self.vocab.index("[unused2]")

            block_text = block_text[:e_offset_orig] + [e_start_marker] + block_text[e_offset_orig:e_end] + \
            [e_end_marker] + block_text[e_end:] + [self.vocab.index("[SEP]")]
            e_offset = 1 + e_offset_orig

            block_text = torch.LongTensor(block_text)
            sent, segment = self.prepend_cls(block_text)

        elif self.use_sep:
            orig_sent_len = len(block_text)
            block_text = block_text + [self.vocab.index("[SEP]")] + block_text[e_offset_orig:e_end] + [self.vocab.index("[SEP]")]
            e_offset = 1 + e_offset_orig
            block_text = torch.LongTensor(block_text)
            sent, segment = self.prepend_cls(block_text)
            segment[orig_sent_len + 2:] = 1

        if self.use_marker:
            assert self.debinarize_list(sent.tolist())[
                e_offset] == '[unused1]'

        # truncate the sample
        item_len = sent.size(0)
        if item_len > self.max_length:
            sent = sent[:self.max_length]
            segment = segment[:self.max_length]
            e_offset = min(e_offset, self.max_length - 1)

        return {'text': sent, 'segment': segment, 'target': type_label, 'e_offset': e_offset}

    def prepend_cls(self, sent):
        cls = sent.new_full((1,), self.vocab.cls())
        sent = torch.cat([cls, sent])
        segment = torch.zeros(sent.size(0)).long()
        return sent, segment

    def debinarize_list(self, indices):
        return [self.vocab[idx] for idx in indices]

    def binarize_list(self, words):
        """
        binarize tokenized sequence
        """
        return [self.vocab.index(w) for w in words]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return collate(samples, self.vocab.pad(), self.num_class)

    def get_dummy_batch(self, num_tokens, max_positions, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        pass

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index] + 3

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index] + 1

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            return np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices

    def prefetch(self, indices):
        self.dataset.prefetch(indices)

    @property
    def supports_prefetch(self):
        return (
            hasattr(self.dataset, 'supports_prefetch')
            and self.dataset.supports_prefetch
        )
