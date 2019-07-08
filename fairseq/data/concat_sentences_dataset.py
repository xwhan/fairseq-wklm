# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

from . import FairseqDataset

MAX_QA = 128
MAX_SEQ = 512

class ConcatSentencesDataset(FairseqDataset):

    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets
        assert all(len(ds) == len(datasets[0]) for ds in datasets), \
            'datasets must have the same length'

    def __getitem__(self, index):
        context = self.datasets[0][index]
        qa = self.datasets[1][index]
        qa_len = qa.size(0)
        if qa_len > MAX_QA:
            qa = qa[:MAX_QA]
        qa_len = qa.size(0)
        if qa_len + context.size(0) > MAX_SEQ:
            context = context[: MAX_SEQ - qa_len]
        return torch.cat([context, qa])

    def __len__(self):
        return len(self.datasets[0])

    def collater(self, samples):
        return self.datasets[0].collater(samples)

    @property
    def sizes(self):
        return sum(ds.sizes for ds in self.datasets)

    def num_tokens(self, index):
        return sum(ds.num_tokens(index) for ds in self.datasets)

    def size(self, index):
        return sum(ds.size(index) for ds in self.datasets)

    def ordered_indices(self):
        return self.datasets[0].ordered_indices()

    @property
    def supports_prefetch(self):
        return any(
            getattr(ds, 'supports_prefetch', False) for ds in self.datasets
        )

    def prefetch(self, indices):
        for ds in self.datasets:
            if getattr(ds, 'supports_prefetch', False):
                ds.prefetch(indices)
