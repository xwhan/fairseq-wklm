# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from functools import reduce
import itertools
import numpy as np
import os
import torch

from torch.utils.data import ConcatDataset

from fairseq.data import (
    REDataset, TokenBlockDataset,
    IndexedDataset)
from fairseq.meters import ClassificationMeter

from . import FairseqTask, register_task

from fairseq.data.masked_lm_dictionary import BertDictionary

from fairseq.tokenization import BertTokenizer


@register_task('re')
class RETask(FairseqTask):
    """
    Classify a sentence

    Args:
        dictionary (Dictionary): the dictionary for the input of the classifier

    The sentence classification task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.sentence_classification_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='path to  data directory', default='/private/home/xwhan/dataset/tacred')
        parser.add_argument('--max-length', type=int, default=512)
        parser.add_argument('--num-class', type=int, default=42)
        parser.add_argument('--use-kdn', action="store_true")

        # kdn parameters
        parser.add_argument('--use-mlm', action='store_true', help='whether add MLM loss for multi-task learning')

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.tokenizer = BertTokenizer(os.path.join(args.data, 'vocab.txt'))
        self.ignore_index = -1
        self.max_length = args.max_length

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        # dictionary = BertDictionary.load(os.path.join(args.data, 'dict.txt'))
        dictionary = BertDictionary.load(os.path.join(args.data, 'dict.txt'))
        print('| get dictionary: {} types from {}'.format(len(dictionary), os.path.join(args.data, 'dict.txt')))

        return cls(args, dictionary)

    def load_dataset(self, split, combine=False, epoch=0):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        loaded_datasets = [[]]
        stop = False
        binarized_data_path = os.path.join(self.args.data, "binarized")
        tokenized_data_path = os.path.join(self.args.data, "processed-splits")
        
        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')

            path_sent = os.path.join(binarized_data_path, split_k, split_k)

            for path, datasets in zip([path_sent], loaded_datasets):
                if IndexedDataset.exists(path):
                    ds = IndexedDataset(path, fix_lua_indexing=True)
                else:
                    if k > 0:
                        stop = True
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({}) {}'.format(split, self.args.data, path))
                datasets.append(
                    TokenBlockDataset(
                        ds, ds.sizes, 0, pad=self.dictionary.pad(), eos=self.dictionary.eos(),
                        break_mode='eos', include_targets=False,
                    ))

            if stop:
                break

            # load start and end labels
            raw_path = os.path.join(tokenized_data_path, split_k)
            e1_offsets = []
            with open(os.path.join(raw_path, 'e1_start.txt'), 'r') as lbl_f:
                lines = lbl_f.readlines()
                for line in lines:
                    lbl = int(line.strip())
                    e1_offsets.append(lbl)

            e2_offsets = []
            with open(os.path.join(raw_path, 'e2_start.txt'), 'r') as lbl_f:
                lines = lbl_f.readlines()
                for line in lines:
                    lbl = int(line.strip())
                    e2_offsets.append(lbl)

            e1_lens = []
            with open(os.path.join(raw_path, 'e1_len.txt'), 'r') as lbl_f:
                lines = lbl_f.readlines()
                for line in lines:
                    lbl = int(line.strip())
                    e1_lens.append(lbl)

            e2_lens = []
            with open(os.path.join(raw_path, 'e2_len.txt'), 'r') as lbl_f:
                lines = lbl_f.readlines()
                for line in lines:
                    lbl = int(line.strip())
                    e2_lens.append(lbl)

            loaded_labels = []
            with open(os.path.join(raw_path, 'lbl.txt'), 'r') as lbl_f:
                lines = lbl_f.readlines()
                for line in lines:
                    lbl = int(line.strip())
                    loaded_labels.append(lbl)

            print('| {} {} {} examples'.format(self.args.data, split_k, len(loaded_datasets[0][-1])))

            if not combine:
                break


        if len(loaded_datasets[0]) == 1:
            dataset = loaded_datasets[0][0]
            sizes = dataset.sizes
        else:
            dataset = ConcatDataset(loaded_datasets[0])
            sizes = np.concatenate([ds.sizes for ds in loaded_datasets[0]])

        assert len(dataset) == len(loaded_labels)

        shuffle = True if split == 'train' else False

        self.datasets[split] = REDataset(
            dataset, loaded_labels, e1_offsets, e1_lens, e2_offsets, e2_lens, sizes, self.dictionary,
            self.args.max_length, shuffle
        )

    def extra_meters(self):
        return {
            'classification': ClassificationMeter(),
        }

    def aggregate_extra_metrics(self, logs):
        return {
            'classification': tuple(
                reduce(lambda q, w: (sum(x) for x in zip(q, w)), [log['extra_metrics']['classification'] for log in logs if 'extra_metrics' in log]))
        }


    def get_loss(self, model, criterion, sample, is_valid=False):
        outputs = criterion(model, sample)

        if is_valid:
            loss, sample_size, logging_output = outputs
            lprobs = logging_output['lprobs']
            pred = torch.argmax(lprobs, dim=-1)
            t = sample['target'].squeeze(-1)
            tp = t.eq(pred).long().sum().item()
            tn = 0
            fp = t.size(0) - tp
            fn = 0

            logging_output['extra_metrics'] = {}
            logging_output['extra_metrics']['classification'] = (tp, tn, fp, fn)

        else:
            loss, sample_size, logging_output = outputs

        return loss, sample_size, logging_output

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary
