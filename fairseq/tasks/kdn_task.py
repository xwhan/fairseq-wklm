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

from torch.utils.data import ConcatDataset

from fairseq.data import (
    KDNDataset, TokenBlockDataset,
    IndexedDataset)
from fairseq.meters import ClassificationMeter

from . import FairseqTask, register_task

from fairseq.data.masked_lm_dictionary import BertDictionary
from fairseq.tokenization import BertTokenizer


@register_task('kdn')
class KDNTask(FairseqTask):
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
        parser.add_argument('data', help='path to  data directory', default='/private/home/xwhan/dataset/webq_qa')
        parser.add_argument('--max-length', type=int, default=64)
        parser.add_argument('--num-labels', type=int, default=2, help='number of labels')
        parser.add_argument('--ignore-index', type=int, default=-1)

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.ignore_index = -1
        self.tokenizer = BertTokenizer(os.path.join(args.data, 'vocab.txt'))

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

    def load_dataset(self, split, combine=False):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        loaded_datasets = [[]]
        stop = False

        binarized_data_path = os.path.join(self.args.data, "binarized")
        tokenized_data_path = os.path.join(self.args.data, "processed-splits")
        
        ent_offsets = []
        ent_lens = []
        ent_lbls = []

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')

            path_context = os.path.join(binarized_data_path, 'context', split_k)
            for path, datasets in zip([path_context], loaded_datasets):
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
                        ds, ds.sizes, 0, pad=self.dictionary.pad(), eos=self.dictionary.eos(), break_mode='eos', include_targets=False,
                    ))

            if stop:
                break

            # load start and end labels
            raw_path = os.path.join(tokenized_data_path, split_k)
            
            with open(os.path.join(raw_path, 'offset.txt'), 'r') as offset_f:
                lines = offset_f.readlines()
                for line in lines:
                    offsets = [int(x) for x in line.strip().split()]
                    ent_offsets.append(offsets)
            
            with open(os.path.join(raw_path, 'len.txt'), 'r') as len_f:
                lines = len_f.readlines()
                for line in lines:
                    lens = [int(x) for x in line.strip().split()]
                    ent_lens.append(lens)
            
            with open(os.path.join(raw_path, 'lbl.txt'), 'r') as lbl_f:
                lines = lbl_f.readlines()
                for line in lines:
                    lbls = [int(x) for x in line.strip().split()]
                    ent_lbls.append(lbls)

            print('| {} {} {} examples'.format(self.args.data, split_k, len(loaded_datasets[0][-1])))

            if not combine:
                break

        if len(loaded_datasets[0]) == 1:
            dataset = loaded_datasets[0][0]
            sizes = dataset.sizes
        else:
            dataset = ConcatDataset(loaded_datasets[0])
            sizes = np.concatenate([ds.sizes for ds in loaded_datasets[0]])

        max_num_ent = max([len(_) for _ in ent_offsets]) # max num of entitites per block

        assert len(dataset) == len(ent_lbls)
        shuffle = True if split == 'train' else False
        self.datasets[split] = KDNDataset(
            dataset, ent_lbls, ent_offsets, ent_lens, sizes, self.dictionary,
            self.args.max_length, max_num_ent, shuffle
        )

    def extra_meters(self):
        return {
            'classification': ClassificationMeter(),
        }

    def aggregate_extra_metrics(self, logs):
        return {
            'classification': tuple(
                reduce(lambda q, w: (sum(x) for x in zip(q, w)), [log['extra_metrics']['classification'] for log in logs if 'extra_metrics' in log])),
            'misclassified': sum([log['extra_metrics']['misclassified'] for log in logs if 'extra_metrics' in log], [])
        }

    def get_loss(self, model, criterion, sample, is_valid=False):
        loss, sample_size, logging_output = criterion(model, sample)

        if is_valid:
            probs = logging_output['lprobs'].exp()
            pos = sample['target'].view(-1).eq(1)
            neg = sample['target'].view(-1).eq(0)

            correct_pos = probs[pos] > 0.5
            correct_neg = probs[neg] > 0.5

            tp = correct_pos.long().sum()
            tn = correct_neg.long().sum()
            fp = neg.long().sum() - tn
            fn = pos.long().sum() - tp

            logging_output['extra_metrics'] = {
                'classification': (tp.item(), tn.item(), fp.item(), fn.item()),
                'misclassified': []
            }

            loss = loss.sum()
            logging_output['loss'] = loss.item()

        return loss, sample_size, logging_output


    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary
