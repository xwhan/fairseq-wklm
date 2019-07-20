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

    target = torch.cat([s['target'].unsqueeze(0) for s in samples], dim=0)
    
    masks = torch.zeros(batch_text.size(0), len(samples[0]['target']), batch_text.size(-1))

    for idx, s in enumerate(samples):
        offsets = s['ent_offsets']
        lens = s['ent_lens']
        for idx_, (offset, length) in enumerate(zip(offsets, lens)):
            masks[idx, idx_, offset:offset+length] = 1 / length

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


class KDNDataset(FairseqDataset):
    """

    Args:
        dataset (torch.utils.data.Dataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
    """

    def __init__(self, dataset, ent_labels, offsets, ent_lens, sizes, dictionary, max_length, max_num_ent, shuffle=False):
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.ent_offsets = offsets
        self.ent_lens = ent_lens
        self.ent_labels = ent_labels
        self.vocab = dictionary
        self.shuffle = shuffle
        self.max_length = max_length
        self.max_num_ent = max_num_ent

    def __getitem__(self, index):
        block_text = self.dataset[index]
        ent_labels = self.ent_labels[index]
        ent_lens = self.ent_lens[index]
        ent_offsets = np.array(self.ent_offsets[index]) + 1 # for cls

        sent, segment = self.prepend_cls(block_text)
        ent_labels_padded = ent_labels + [-1] * (self.max_num_ent - len(ent_labels))
        ent_labels_padded = torch.tensor(ent_labels_padded).long()

        return {'text': sent, 'segment': segment, 'target': ent_labels_padded, 'ent_offsets': ent_offsets, "ent_lens": ent_lens}

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
        # if isinstance(max_positions, float) or isinstance(max_positions, int):
        #     tgt_len = min(tgt_len, max_positions)
        # bsz = num_tokens // tgt_len
        # sent1 = self.vocab.dummy_sentence(tgt_len + 2)
        # sent2 = self.vocab.dummy_sentence(tgt_len + 2)

        # sent1[sent1.eq(self.vocab.unk())] = 66
        # sent2[sent2.eq(self.vocab.unk())] = 66
        # text, segment = self._join_sents(sent1, sent2)

        # paragraph_mask = torch.zeros(text.shape).byte()
        # paragraph_mask[sent2.numel():] = 1
        
        # target = (torch.tensor([self.vocab.pad()]), torch.tensor([self.vocab.pad()]))
        # idx_map = [self.vocab.pad()]
        # token_is_max_context = [0]
        # return self.collater([
        #     {'id': i, 'text': text, 'target': target, 'segment': segment, 'paragraph_mask': paragraph_mask, 'squad_ids': 0, 'actual_txt':'dummy', 'idx_map':idx_map,'token_is_max_context':token_is_max_context}
        #     for i in range(bsz)
        # ])

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

        # indices = indices[np.argsort(self.sizes1[indices], kind='mergesort')]
        # return indices[np.argsort(self.sizes2[indices], kind='mergesort')]

    def prefetch(self, indices):
        self.dataset.prefetch(indices)

    @property
    def supports_prefetch(self):
        return (
                hasattr(self.dataset, 'supports_prefetch')
                and self.dataset.supports_prefetch
        )
