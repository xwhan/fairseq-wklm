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
from fairseq.meters import F1Meter

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
        parser.add_argument('--use-hf', action="store_true")
        parser.add_argument('--use-marker', action="store_true")
        parser.add_argument('--use-ner', action='store_true')
        parser.add_argument('--use-cased', action='store_true')
        parser.add_argument('--last-drop', type=float, default=0.0, help='dropout before projection')
        parser.add_argument('--no-rel-id', type=int, default=1)
        parser.add_argument('--final-metric', type=str, default=None, help="metric for model selection")

        # kdn parameters
        parser.add_argument('--use-mlm', action='store_true', help='whether add MLM loss for multi-task learning')
        parser.add_argument("--add-layer", action='store_true')

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.tokenizer = BertTokenizer(os.path.join(args.data, 'vocab.txt'))
        self.ignore_index = -1
        self.max_length = args.max_length
        self.use_marker = args.use_marker
        self.use_ner = args.use_ner
        self.use_cased = args.use_cased
        self.no_rel_id = args.no_rel_id
        self.final_metric = args.final_metric

        assert not (self.use_marker and self.use_ner)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        # dictionary = BertDictionary.load(os.path.join(args.data, 'dict.txt'))
        if args.use_cased:
            dictionary = BertDictionary.load("/private/home/xwhan/fairseq-py/vocab_cased/dict.txt")
        else:
            dictionary = BertDictionary.load("/private/home/xwhan/fairseq-py/vocab_dicts/dict.txt")
        print('| get dictionary: {} types from {}'.format(len(dictionary), os.path.join(args.data, 'dict.txt')))

        return cls(args, dictionary)

    def load_dataset(self, split, combine=False, epoch=0):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        loaded_datasets = [[]]
        stop = False

        if self.use_cased:
            binarized_data_path = os.path.join(self.args.data, "binarized-cased")
            tokenized_data_path = os.path.join(self.args.data, "processed-splits-cased")
        else:
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

            e1_type = []
            with open(os.path.join(raw_path, 'e1_type.txt'), 'r') as lbl_f:
                lines = lbl_f.readlines()
                for line in lines:
                    lbl = line.strip()
                    e1_type.append(lbl)

            e2_type = []
            with open(os.path.join(raw_path, 'e2_type.txt'), 'r') as lbl_f:
                lines = lbl_f.readlines()
                for line in lines:
                    lbl = line.strip()
                    e2_type.append(lbl)

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
            dataset, loaded_labels, e1_offsets, e1_lens, e2_offsets, e2_lens, e1_type, e2_type, sizes, self.dictionary,
            self.args.max_length, shuffle, use_ner=self.use_ner, use_marker=self.use_marker
        )

    def extra_meters(self):
        return {
            'F1': F1Meter(),
        }

    def aggregate_extra_metrics(self, logs):
        return {
            'F1': tuple(
                reduce(lambda q, w: (sum(x) for x in zip(q, w)), [log['extra_metrics']['F1'] for log in logs if 'extra_metrics' in log]))
        }


    def get_loss(self, model, criterion, sample, is_valid=False):
        outputs = criterion(model, sample)

        if is_valid:
            loss, sample_size, logging_output = outputs
            lprobs = logging_output['lprobs']
            pred = torch.argmax(lprobs, dim=-1).tolist()
            target = sample['target'].squeeze(-1).tolist()

            n_gold = n_pred = n_corr = 0
            for p, t in zip(pred, target):
                if p != self.no_rel_id:
                    n_pred += 1
                if t != self.no_rel_id:
                    n_gold += 1
                if (p != self.no_rel_id) and (t != self.no_rel_id) and (p == t):
                    n_corr +=1

            logging_output['extra_metrics'] = {"F1": (n_pred, n_gold, n_corr)}

        else:
            loss, sample_size, logging_output = outputs

        return loss, sample_size, logging_output

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary
