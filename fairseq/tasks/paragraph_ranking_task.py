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
import json
from tqdm import tqdm


from fairseq.data import (
    Dictionary, IndexedCachedDataset, IndexedRawTextDataset,
    ParagraphRankingDataset, TokenBlockDataset,
    IndexedDataset)
from fairseq.meters import ClassificationMeter

from . import FairseqTask, register_task
from fairseq.data.masked_lm_dictionary import BertDictionary

from fairseq.tokenization import BertTokenizer


@register_task('paragaph_ranking')
class ParagraphRankingTask(FairseqTask):
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
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--num-labels', type=int, default=2,
                            help='number of labels')
        parser.add_argument('--lazy-load', action='store_true', help='load the dataset lazily')

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.padding_idx = -100
        self.num_labels = args.num_labels
        self.tokenizer = BertTokenizer(os.path.join(args.data, 'vocab.txt'))

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).
        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dictionary = BertDictionary.load(os.path.join(args.data, 'dict.txt'))
        print('| dictionary: {} types'.format(len(dictionary)))

        return cls(args, dictionary)

    def load_dataset(self, split, combine=False):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        raw_data = open(os.path.join(self.args.data, '{}.json'.format(split))).readlines()
        loaded_labels = []
        dataset = []
        sizes = []
        print('Processing raw {} data ...'.format(split))

        for item in tqdm(raw_data):
            item = json.loads(item)
            ques_sent = item['q']
            para_sent = item['para']
            label = int(item['label'])

            loaded_labels.append(label)

            ques_sent_idx = self.dictionary.encode_line(ques_sent.lower(), line_tokenizer=self.tokenizer.tokenize, append_eos=False, add_if_not_exist=False)
            para_sent_idx = self.dictionary.encode_line(para_sent.lower(), line_tokenizer=self.tokenizer.tokenize, append_eos=False, add_if_not_exist=False)

            if len(para_sent_idx.tolist() + ques_sent_idx.tolist()) + 3 > 512:
                extra_len = len(para_sent_idx.tolist() + ques_sent_idx.tolist()) + 3 - 512
                para_sent_idx = para_sent_idx[:-extra_len]

            dataset.append({'q': ques_sent_idx, 'para': para_sent_idx})
            sizes.append(len(para_sent_idx.tolist() + ques_sent_idx.tolist()))


        self.datasets[split] = ParagraphRankingDataset(
            dataset, loaded_labels, sizes, self.dictionary, True if split == 'train' else False
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
        loss, sample_size, logging_output = criterion(model, sample, reduce=not is_valid)

        if is_valid:
            assert self.num_labels == 2
            probs = (-loss).exp()
            pos = sample['target'].view(-1).eq(1)
            neg = sample['target'].view(-1).eq(0)

            correct_pos = probs[pos] > 1 / self.num_labels
            correct_neg = probs[neg] > 1 / self.num_labels

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

            if False:
                correct = pos.new_zeros(pos.shape)
                correct[pos] = correct_pos
                correct[neg] = correct_neg
                incorrect = ~correct
                incorrect_ids = sample['id'][incorrect.nonzero()]
                logging_output['extra_metrics']['misclassified'] = incorrect_ids.squeeze().tolist()

        return loss, sample_size, logging_output

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary
