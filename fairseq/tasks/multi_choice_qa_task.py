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
    MultiChoiceQADataset, TokenBlockDataset,
    IndexedDataset)
from fairseq.meters import ClassificationMeter

from . import FairseqTask, register_task
from fairseq.data.masked_lm_dictionary import BertDictionary

from fairseq.tokenization import BertTokenizer


@register_task('multi_choice_qa')
class MultiChoiceQATask(FairseqTask):
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

        raw_data = json.load(open(os.path.join(self.args.data, '{}.json'.format(split))))
        loaded_labels = []
        dataset = []
        sizes = []
        print('Processing raw {} data ...'.format(split))

        for item in raw_data:
            for choice in item['choices']:
                refer_doc_sent = choice['refer_doc']
                ques_sent = choice['question']
                choice_sent = choice['choice']
                label = choice['correct']

                refer_doc_idx = self.dictionary.encode_line(refer_doc_sent, line_tokenizer=self.tokenizer.tokenize, append_eos=False)
                ques_idx = self.dictionary.encode_line(ques_sent, line_tokenizer=self.tokenizer.tokenize, append_eos=False)
                choice_idx = self.dictionary.encode_line(choice_sent, line_tokenizer=self.tokenizer.tokenize, append_eos=False)

                 # 4 for special tokens
                if len(refer_doc_idx.tolist() + ques_idx.tolist() + choice_idx.tolist()) + 4 > 512:
                    extra_len = len(refer_doc_idx.tolist() + ques_idx.tolist() + choice_idx.tolist()) + 4 - 512
                    refer_doc_idx = refer_doc_idx[:-extra_len]

                dataset.append({"refer": refer_doc_idx, "q": ques_idx, "choice": choice_idx})
                loaded_labels.append(label)
                sizes.append(len(refer_doc_idx.tolist() + ques_idx.tolist() + choice_idx.tolist()))

        self.datasets[split] = MultiChoiceQADataset(
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
