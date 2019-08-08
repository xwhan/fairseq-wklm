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
    SpanQADataset, TokenBlockDataset,
    IndexedDataset)
from fairseq.meters import ClassificationMeter

from . import FairseqTask, register_task

from fairseq.data.masked_lm_dictionary import BertDictionary

from fairseq.tokenization import BertTokenizer


@register_task('span_qa')
class SpanQATask(FairseqTask):
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
        parser.add_argument('--max-length', type=int, default=512)
        parser.add_argument('--max-query-length', type=int, default=18)
        parser.add_argument('--use-kdn', action="store_true")

        # kdn parameters
        parser.add_argument('--use-mlm', action='store_true', help='whether add MLM loss for multi-task learning')

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.valid_groups = ('classification_start', 'classification_end')
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

    def load_dataset(self, split, combine=False, epoch=0):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        loaded_datasets = [[], []]
        loaded_labels = []
        loaded_ids = []
        loaded_raw_question_text = []
        loaded_raw_context_text = []
        stop = False

        binarized_data_path = os.path.join(self.args.data, "binarized")
        tokenized_data_path = os.path.join(self.args.data, "processed-splits")
        
        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')

            path_q = os.path.join(binarized_data_path, 'q', split_k)
            path_c = os.path.join(binarized_data_path, 'c', split_k)

            for path, datasets in zip([path_q, path_c], loaded_datasets):
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
            starts = []
            with open(os.path.join(raw_path, 'ans_start.txt'), 'r') as lbl_f:
                lines = lbl_f.readlines()
                for line in lines:
                    lbls = [int(x) for x in line.strip().split()]
                    starts.append(lbls[0])
            ends = []
            with open(os.path.join(raw_path, 'ans_end.txt'), 'r') as lbl_f:
                lines = lbl_f.readlines()
                for line in lines:
                    lbls = [int(x) for x in line.strip().split()]
                    ends.append(lbls[0])
            loaded_labels = list(zip(starts, ends))

            with open(os.path.join(raw_path, 'q.txt'), 'r') as act_f:
                lines = act_f.readlines()
                for line in lines:
                    loaded_raw_question_text.append(line.strip())

            with open(os.path.join(raw_path, 'c.txt'), 'r') as act_f:
                lines = act_f.readlines()
                for line in lines:
                    loaded_raw_context_text.append(line.strip())

            with open(os.path.join(raw_path, 'sample_id.txt'), 'r') as id_f:
                loaded_ids.extend([id.strip() for id in id_f.readlines()])

            print('| {} {} {} examples'.format(self.args.data, split_k, len(loaded_datasets[0][-1])))

            if not combine:
                break

        if len(loaded_datasets[0]) == 1:
            dataset_q = loaded_datasets[0][0]
            dataset_c = loaded_datasets[1][0]
            sizes_q = dataset_q.sizes
            sizes_c = dataset_c.sizes
        else:
            dataset_q = ConcatDataset(loaded_datasets[0])
            dataset_c = ConcatDataset(loaded_datasets[1])
            sizes_q = np.concatenate([ds.sizes for ds in loaded_datasets[0]])
            sizes_c = np.concatenate([ds.sizes for ds in loaded_datasets[1]])

        assert len(dataset_q) == len(loaded_labels)
        assert len(dataset_c) == len(loaded_labels)

        shuffle = True if split == 'train' else False

        self.datasets[split] = SpanQADataset(
            dataset_q, dataset_c, loaded_labels, 
            loaded_ids, loaded_raw_question_text, loaded_raw_context_text, sizes_q, 
            sizes_c, self.dictionary,
            self.args.max_length, self.args.max_query_length, shuffle
        )

    def extra_meters(self):
        return {
            'classification_start': ClassificationMeter('start'),
            'classification_end': ClassificationMeter('end'),
        }

    def aggregate_extra_metrics(self, logs):
        agg = {}
        for m in self.valid_groups:
            agg[m] = tuple(
                reduce(lambda q, w: (sum(x) for x in zip(q, w)),
                       [log['extra_metrics'][m] for log in logs if 'extra_metrics' in log]))
        return agg

    def get_loss(self, model, criterion, sample, is_valid=False):
        outputs = criterion(model, sample, is_valid=is_valid)

        if is_valid:
            loss, sample_size, logging_output, outs = outputs
            logging_output['extra_metrics'] = {}
            for g, o, t in zip(self.valid_groups, outs, sample['target']):
                t = t.squeeze(-1)
                pred_t = torch.argmax(o, dim=-1)
                tp = t.eq(pred_t).long().sum().item()
                tn = 0
                fp = t.size(0) - tp
                fn = 0
                logging_output['extra_metrics'][g] = (tp, tn, fp, fn)
        else:
            loss, sample_size, logging_output = outputs

        return loss, sample_size, logging_output

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary
