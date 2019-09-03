# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch
import math
from typing import Tuple

from . import data_utils, FairseqDataset

class KDNDataset(FairseqDataset):
    """

    Args:
        dataset (torch.utils.data.Dataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
    """

    def __init__(
        self, 
        dataset, 
        ent_labels, 
        offsets, 
        ent_lens, 
        sizes, 
        dictionary, 
        max_length,
        max_num_ent, 
        shuffle=False,
        seed=1,
        masking_ratio=0.15,
        masking_prob=0.8,
        random_token_prob=0.1,
        segment_id=0,
        use_mlm=False,
    ):
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.ent_offsets = offsets
        self.ent_lens = ent_lens
        self.ent_labels = ent_labels
        self.vocab = dictionary
        self.shuffle = shuffle
        self.max_length = max_length
        self.max_num_ent = max_num_ent
        self.seed = seed
        self.masking_ratio = masking_ratio
        self.masking_prob = masking_prob
        self.random_token_prob = random_token_prob
        self.segment_id = segment_id
        self.use_mlm = use_mlm

    def __getitem__(self, index):
        block_text = self.dataset[index]
        ent_labels = self.ent_labels[index]
        ent_lens = self.ent_lens[index]
        ent_offsets = self.ent_offsets[index]
        assert len(ent_labels) == len(ent_lens) == len(ent_offsets)
        ent_labels_padded = ent_labels + [-1] * (self.max_num_ent - len(ent_labels))
        ent_labels_padded = torch.tensor(ent_labels_padded).long()

        return {'id': index, 'block': block_text, 'target': ent_labels_padded, 'ent_offsets': ent_offsets, "ent_lens": ent_lens}

    def collate(self, samples, pad_idx):

        if len(samples) == 0:
            return {}
        if not isinstance(samples[0], dict):
            samples = [s for sample in samples for s in sample]

        with data_utils.numpy_seed(self.seed + samples[0]["id"]):
            for s in samples:
                if self.use_mlm:
                    token_range = (self.vocab.nspecial, len(self.vocab))
                    masked_blk_one, masked_tgt_one = self._mask_block(s["block"], self.vocab.mask(), self.vocab.pad(), token_range, (s['ent_offsets'], s['ent_lens']))
                    tokens = np.concatenate([[self.vocab.cls()], masked_blk_one, [self.vocab.sep()]])
                    segments = np.ones(len(tokens)) * self.segment_id
                    targets = np.concatenate([[self.vocab.pad()], masked_tgt_one, [self.vocab.pad()]])
                    s['sent'] = torch.LongTensor(tokens)
                    s['segment'] = torch.LongTensor(segments)
                    s["lm_target"] = torch.LongTensor(targets)
                else:
                    tokens = np.concatenate([[self.vocab.cls()], s['block'], [self.vocab.sep()]])
                    segments = np.ones(len(tokens)) * self.segment_id
                    s['sent'] = torch.LongTensor(tokens)
                    s['segment'] = torch.LongTensor(segments)
                    s['lm_target'] = None
                s['ent_offsets'] = [ii+1 for ii in s['ent_offsets']] # for cls

        batch_text = data_utils.collate_tokens([s['sent'] for s in samples], pad_idx, left_pad=False)
        batch_seg = data_utils.collate_tokens([s['segment'] for s in samples], pad_idx, left_pad=False)
        batch_lm_target = data_utils.collate_tokens([s['lm_target'] for s in samples], pad_idx, left_pad=False) if self.use_mlm else None

        target = torch.cat([s['target'].unsqueeze(0) for s in samples], dim=0)
        masks = torch.zeros(batch_text.size(0), len(samples[0]['target']), batch_text.size(-1))

        for idx, s in enumerate(samples):
            offsets = s['ent_offsets']
            lens = s['ent_lens']
            for idx_, (offset, length) in enumerate(zip(offsets, lens)):
                if length == 0:
                    target[idx, idx_] = -1
                    continue

                # average mask
                # masks[idx, idx_, offset:offset+length] = 1 / length

                # only use the start and end tok of of the entity
                masks[idx, idx_, offset] = 1
                masks[idx, idx_, offset+length-1] = 2

                # mask of entity boundaries
                masks[idx, idx_, offset - 1] = -1
                masks[idx, idx_, offset+length] = -2

        return {
            'ntokens': sum(len(s['sent']) for s in samples),
            'net_input': {
                'sentence': batch_text,
                'segment': batch_seg,
                'entity_masks': masks
            },
            'target': target,
            'lm_target': batch_lm_target,
            'nsentences': len(samples),
        }

    def _mask_block(
            self,
            sentence: np.ndarray,
            mask_idx: int,
            pad_idx: int,
            dictionary_token_range: Tuple,
            entity_info: Tuple,
    ):
        """
        Mask tokens for Masked Language Model training
        Samples mask_ratio tokens that will be predicted by LM.

        Note:This function may not be efficient enough since we had multiple
        conversions between np and torch, we can replace them with torch
        operators later.

        Args:
            sentence: 1d tensor to be masked
            mask_idx: index to use for masking the sentence
            pad_idx: index to use for masking the target for tokens we aren't
                predicting
            dictionary_token_range: range of indices in dictionary which can
                be used for random word replacement
                (e.g. without special characters)
            entity_info: entity offsets and lens of masked entities, added to prevent masking of entity tokens
        Return:
            masked_sent: masked sentence
            target: target with words which we are not predicting replaced
                by pad_idx
        """
        masked_sent = np.copy(sentence)
        sent_length = len(sentence)
        mask_num = math.ceil(sent_length * self.masking_ratio)
        mask = np.random.choice(sent_length, mask_num, replace=False)
        target = np.copy(sentence)

        # entity token index
        entity_tok_index = []
        for offset, len_ in zip(entity_info[0], entity_info[1]):
            for _ in range(len_):
                entity_tok_index.append(offset + _)

        for i in range(sent_length):
            if i in mask and (i not in entity_tok_index):
                rand = np.random.random()

                # replace with mask if probability is less than masking_prob
                # (Eg: 0.8)
                if rand < self.masking_prob:
                    masked_sent[i] = mask_idx

                # replace with random token if probability is less than
                # masking_prob + random_token_prob (Eg: 0.9)
                elif rand < (self.masking_prob + self.random_token_prob):
                    # sample random token from dictionary
                    masked_sent[i] = (
                        np.random.randint(
                            dictionary_token_range[0], dictionary_token_range[1]
                        )
                    )
            else:
                target[i] = pad_idx

        return masked_sent, target


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
        return self.collate(samples, self.vocab.pad())

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
