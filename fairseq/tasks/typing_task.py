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
    TypingDataset, TokenBlockDataset,
    IndexedDataset)
from fairseq.meters import TypingMeter

from . import FairseqTask, register_task

from fairseq.data.masked_lm_dictionary import BertDictionary

from fairseq.tokenization import BertTokenizer


@register_task('typing')
class TypingTask(FairseqTask):
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
        parser.add_argument('data', help='path to  data directory',
                            default='/private/home/xwhan/dataset/tacred')
        parser.add_argument('--max-length', type=int, default=256)
        parser.add_argument('--num-class', type=int, default=113)
        parser.add_argument('--use-kdn', action="store_true")
        parser.add_argument('--use-marker', action="store_true")
        parser.add_argument('--use-sep', action="store_true")
        parser.add_argument('--thresh', default=0.5, type=float)

        parser.add_argument('--last-drop', type=float,
                            default=0.0, help='dropout before projection')
        parser.add_argument('--final-metric', type=str,
                            default="loss", help="metric for model selection")

        # kdn parameters
        parser.add_argument('--use-mlm', action='store_true',
                            help='whether add MLM loss for multi-task learning')
        parser.add_argument("--add-layer", action='store_true')
        parser.add_argument("--boundary-loss", action='store_true')
        parser.add_argument("--start-end", action='store_true')
        parser.add_argument("--num-kdn", default=4, type=int)
        parser.add_argument("--masking-ratio", default=0.15, type=float)

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.tokenizer = BertTokenizer(os.path.join(args.data, 'vocab.txt'))
        self.ignore_index = -1
        self.max_length = args.max_length
        self.use_marker = args.use_marker
        self.final_metric = args.final_metric
        self.use_sep = args.use_sep
        self.num_class = args.num_class
        self.thresh = args.thresh

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        # dictionary = BertDictionary.load(os.path.join(args.data, 'dict.txt'))

        dictionary = BertDictionary.load(
                "/private/home/xwhan/fairseq-py/vocab_dicts/dict.txt")
        print('| get dictionary: {} types from {}'.format(
            len(dictionary), os.path.join(args.data, 'dict.txt')))

        return cls(args, dictionary)

    def load_dataset(self, split, combine=False, epoch=0):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        loaded_datasets = [[]]
        stop = False

        binarized_data_path = os.path.join(self.args.data, "binarized")
        tokenized_data_path = os.path.join(
                self.args.data, "processed-splits")


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
                        raise FileNotFoundError(
                            'Dataset not found: {} ({}) {}'.format(split, self.args.data, path))
                datasets.append(
                    TokenBlockDataset(
                        ds, ds.sizes, 0, pad=self.dictionary.pad(), eos=self.dictionary.eos(),
                        break_mode='eos', include_targets=False,
                    ))

            if stop:
                break

            # load start and end labels
            raw_path = os.path.join(tokenized_data_path, split_k)
            e_offsets = []
            with open(os.path.join(raw_path, 'e_start.txt'), 'r') as lbl_f:
                lines = lbl_f.readlines()
                for line in lines:
                    lbl = int(line.strip())
                    e_offsets.append(lbl)


            e_lens = []
            with open(os.path.join(raw_path, 'e_len.txt'), 'r') as lbl_f:
                lines = lbl_f.readlines()
                for line in lines:
                    lbl = int(line.strip())
                    e_lens.append(lbl)

            loaded_labels = []
            with open(os.path.join(raw_path, 'lbl.txt'), 'r') as lbl_f:
                lines = lbl_f.readlines()
                for line in lines:
                    lbl = [int(ii) for ii in line.strip().split()]
                    loaded_labels.append(lbl)

            print('| {} {} {} examples'.format(
                self.args.data, split_k, len(loaded_datasets[0][-1])))

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

        self.datasets[split] = TypingDataset(
            dataset, e_offsets, e_lens, loaded_labels, sizes, self.dictionary,
            self.max_length, self.num_class, shuffle, use_marker=self.use_marker, use_sep=self.use_sep
        )

    def extra_meters(self):
        return {
            'Typing': TypingMeter(),
        }

    def aggregate_extra_metrics(self, logs):
        return {
            'Typing': tuple(
                reduce(lambda q, w: (sum(x) for x in zip(q, w)), [log['extra_metrics']['Typing'] for log in logs if 'extra_metrics' in log]))
        }

    def get_loss(self, model, criterion, sample, is_valid=False):
        outputs = criterion(model, sample)

        if is_valid:
            loss, sample_size, logging_output = outputs
            type_probs = logging_output['probs']

            assert sample_size == type_probs.size(0)
            
            n_pred = n_true = n_corr = ma_p = ma_r = strict_acc = 0
            for idx in range(sample_size):
                pred_probs = type_probs[idx,:].tolist()
                target = sample['target'][idx,:]
                true_labels, predicted_labels = [], []

                max_prob = max(pred_probs)
                for type_idx, prob in enumerate(pred_probs):
                    if prob > self.thresh or prob == max_prob:
                        predicted_labels.append(type_idx)
                    if target[type_idx] == 1:
                        true_labels.append(type_idx)

                strict_acc += set(predicted_labels) == set(true_labels)
                n_pred += len(predicted_labels)
                n_true += len(true_labels)
                n_corr += len(set(predicted_labels).intersection(set(true_labels)))

                ma_p += len(set(predicted_labels).intersection(
                    set(true_labels))) / float(len(predicted_labels))
                ma_r += len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
            
            # import pdb;pdb.set_trace()

            logging_output['extra_metrics'] = {"Typing": (n_pred, n_true, n_corr, ma_p, ma_r, strict_acc, sample_size)}

        else:
            loss, sample_size, logging_output = outputs

        return loss, sample_size, logging_output

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary
