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
    target_len = len(samples[0]['target'])
    target = [torch.stack([s['target'][i] for s in samples], dim=0) for i in range(target_len)]
    q_txt = [s['q_txt'] for s in samples]
    c_txt = [s['c_txt'] for s in samples]

    return {
        'id': [s['id'] for s in samples],
        'ntokens': sum(len(s['text']) for s in samples),
        'net_input': {
            'text': data_utils.collate_tokens(
                [s['text'] for s in samples], pad_idx, left_pad=False,
            ),
            'paragraph_mask': data_utils.collate_tokens([s['paragraph_mask'] for s in samples], pad_idx,  left_pad=False),
            'segment': data_utils.collate_tokens(
                [s['segment'] for s in samples], pad_idx, left_pad=False,
            ),
        },
        'target': target,
        'nsentences': len(samples),
        'c_txt': c_txt,
        'q_txt': q_txt,
        'possible_sentences': sum(int(s['target'][0] != -1) for s in samples),
    }


class SpanQADataset(FairseqDataset):
    """

    Args:
        dataset (torch.utils.data.Dataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
    """

    def __init__(self, dataset1, dataset2, 
                       labels, ids, _1_txt, _2_txt, sizes1, sizes2, 
                       dictionary,
                       max_length, max_query_length, shuffle=False):
        self.dataset1, self.dataset2 = dataset1, dataset2
        self.sizes1, self.sizes2 = np.array(sizes1), np.array(sizes2)
        self.labels = np.array(labels)
        self.ids = ids
        self.txt_1 = _1_txt
        self.txt_2 = _2_txt
        self.vocab = dictionary
        self.shuffle = shuffle
        self.max_length = max_length
        self.max_query_length = max_query_length

    def __getitem__(self, index):
        question = self.dataset1[index]
        paragraph = self.dataset2[index]
        lbl = self.labels[index] # (start, end)
        q_txt = self.txt_1[index]
        c_txt = self.txt_2[index]
        sample_id = self.ids[index]
        if question.size(0) > self.max_query_length:
            question = question[:self.max_query_length]
        question_len = question.size(0) + 2  # account for cls + sep

        max_tokens_for_doc = self.max_length - question_len - 1

        if paragraph.size(0) > max_tokens_for_doc:
            paragraph = paragraph[:max_tokens_for_doc]

        if len(lbl) == 0:
            s, e = -1, -1
        else:
            s, e = lbl
            assert e >= s

        if s == -1:
            start_position, end_position = -1, -1 # to be ignored during training
        elif s >= paragraph.size(0):
            start_position, end_position = -1, -1
        else:
            start_position = min(s, paragraph.size(0) - 1) + question_len
            end_position = min(e, paragraph.size(0) - 1) + question_len

        start_position = torch.LongTensor([start_position])
        end_position = torch.LongTensor([end_position])
        text, seg = self._join_sents(question, paragraph)
        paragraph_mask = torch.zeros(text.shape).byte()
        paragraph_mask[question_len : -1] = 1

        target = (start_position, end_position)

        return {'id': sample_id, 'text': text, 'segment': seg, 'target': target, 'paragraph_mask': paragraph_mask, 'q_txt': q_txt, 'c_txt': c_txt}


    def _join_sents(self, sent1, sent2):
        cls = sent1.new_full((1,), self.vocab.cls())
        sep = sent1.new_full((1,), self.vocab.sep())
        sent1 = torch.cat([cls, sent1, sep])
        sent2 = torch.cat([sent2, sep])
        text = torch.cat([sent1, sent2])
        segment1 = torch.zeros(sent1.size(0)).long()
        segment2 = torch.ones(sent2.size(0)).long()
        # segment2 = torch.zeros(sent2.size(0)).long()
        segment = torch.cat([segment1, segment2])
        return text, segment

    def debinarize_list(self, indices):
        return [self.vocab[idx] for idx in indices]

    def binarize_list(self, words):
        """
        binarize tokenized sequence
        """
        return [self.vocab.index(w) for w in words]

    def __len__(self):
        return len(self.dataset1)

    def collater(self, samples):
        return collate(samples, self.vocab.pad())

    def get_dummy_batch(self, num_tokens, max_positions, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            tgt_len = min(tgt_len, max_positions)
        bsz = num_tokens // tgt_len
        sent1 = self.vocab.dummy_sentence(tgt_len + 2)
        sent2 = self.vocab.dummy_sentence(tgt_len + 2)

        sent1[sent1.eq(self.vocab.unk())] = 66
        sent2[sent2.eq(self.vocab.unk())] = 66
        text, segment = self._join_sents(sent1, sent2)

        paragraph_mask = torch.zeros(text.shape).byte()
        paragraph_mask[sent2.numel():] = 1
        
        target = (torch.tensor([self.vocab.pad()]), torch.tensor([self.vocab.pad()]))
        idx_map = [self.vocab.pad()]
        token_is_max_context = [0]
        return self.collater([
            {'id': i, 'text': text, 'target': target, 'segment': segment, 'paragraph_mask': paragraph_mask, 'squad_ids': 0, 'actual_txt':'dummy', 'idx_map':idx_map,'token_is_max_context':token_is_max_context}
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes1[index] + self.sizes2[index] + 3

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes1[index] + self.sizes2[index] + 3

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
        self.dataset1.prefetch(indices)
        self.dataset2.prefetch(indices)

    @property
    def supports_prefetch(self):
        return (
                hasattr(self.dataset1, 'supports_prefetch')
                and self.dataset1.supports_prefetch
                and hasattr(self.dataset2, 'supports_prefetch')
                and self.dataset2.supports_prefetch
        )
