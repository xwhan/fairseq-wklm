# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from . import data_utils, FairseqDataset


def collate(samples, pad_idx):

    if len(samples) == 0:
        return {}
    if not isinstance(samples[0], dict):
        samples = [s for sample in samples for s in sample]

    batch_text = data_utils.collate_tokens([s['text'] for s in samples], pad_idx, left_pad=False)

    target = torch.tensor([s['target'] for s in samples])
    target = target.unsqueeze(1)

    masks = torch.zeros(batch_text.size(0), batch_text.size(1))

    for idx, s in enumerate(samples):
        e1_offset = s['e1_offset']
        e2_offset = s['e2_offset']

        masks[idx, e1_offset] = 1
        masks[idx, e2_offset] = 2

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


class REDataset(FairseqDataset):
    """
    Args:
        dataset (torch.utils.data.Dataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
    """

    def __init__(self, dataset, rel_labels, e1_offsets, e1_lens, e2_offsets, e2_lens, sizes, dictionary, max_length, shuffle=False, use_marker=False):
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.e1_offsets = e1_offsets
        self.e1_lens = e1_lens
        self.e2_offsets = e2_offsets
        self.e2_lens = e2_lens
        self.rel_labels = rel_labels
        self.vocab = dictionary
        self.shuffle = shuffle
        self.max_length = max_length
        self.use_marker = use_marker

    def __getitem__(self, index):
        block_text = self.dataset[index]
        rel_label = self.rel_labels[index]
        e1_offset = self.e1_offsets[index]
        e2_offset = self.e2_offsets[index]
        e1_len = self.e1_lens[index]
        e2_len = self.e2_lens[index]

        e1_start_marker = self.vocab.index("[unused0]")
        e1_end_marker = self.vocab.index("[unused1]")
        e2_start_marker = self.vocab.index("[unused2]")
        e2_end_marker = self.vocab.index("[unused3]")

        e1_end = e1_offset + e1_len
        e2_end = e2_offset + e2_len

        block_text = block_text.tolist()

        if self.use_marker:
            if e1_offset < e2_offset:
                assert e1_end <= e2_offset
                block_text = block_text[:e1_offset] + \
                [e1_start_marker] + \
                block_text[e1_offset:e1_end] + \
                [e1_end_marker] +  \
                block_text[e1_end:e2_offset] + \
                [e2_start_marker] + \
                block_text[e2_offset:e2_end] + \
                [e2_end_marker] + \
                block_text[e2_end:]

                e1_offset += 1 
                e2_offset += 3

            else:
                assert e2_end <= e1_offset
                block_text = block_text[:e2_offset] + \
                [e2_start_marker] + \
                block_text[e2_offset:e2_end] + \
                [e2_end_marker] + \
                block_text[e2_end:e1_offset] + \
                [e1_start_marker] + \
                block_text[e1_offset:e1_end] + \
                [e1_end_marker] + \
                block_text[e1_end:]

                e2_offset += 1
                e1_offset += 3

        block_text = torch.LongTensor(block_text)

        sent, segment = self.prepend_cls(block_text)

        # truncate the sample
        item_len = sent.size(0)
        if item_len > self.max_length:
            sent = sent[:self.max_length]
            segment = segment[:self.max_length]
            e1_offset = min(e1_offset, self.max_length - 1)
            e2_offset = min(e2_offset, self.max_length - 1)

        return {'text': sent, 'segment': segment, 'target': rel_label, 'e1_offset': e1_offset, 'e2_offset': e2_offset}

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
        return collate(samples, self.vocab.pad())

    def get_dummy_batch(self, num_tokens, max_positions, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        pass

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index] + 1

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index] + 1

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            return  np.random.permutation(len(self))
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
