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
        parser.add_argument('--max-length', type=int, default=512)
        parser.add_argument('--num-labels', type=int, default=2,
                            help='number of labels')
        parser.add_argument('--use-kdn', action="store_true")
        parser.add_argument('--final-metric', type=str,
                            default="loss", help="metric for model selection")

        # kdn parameters
        parser.add_argument('--use-mlm', action='store_true',
                            help='whether add MLM loss for multi-task learning')
        parser.add_argument("--add-layer", action='store_true')
        parser.add_argument("--start-end", action='store_true')
        parser.add_argument("--boundary-loss", action='store_true')
        parser.add_argument("--num-kdn", default=4, type=int)

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.ignore_index = -100
        self.num_labels = args.num_labels
        self.tokenizer = BertTokenizer(os.path.join(
            args.data, 'vocab.txt'), do_lower_case=True)
        self.final_metric = args.final_metric
        self.max_length = args.max_length

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).
        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dictionary = BertDictionary.load(os.path.join(args.data, 'dict.txt'))
        print('| dictionary: {} types'.format(len(dictionary)))

        return cls(args, dictionary)

    def load_dataset(self, split, combine=False, epoch=0):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        loaded_datasets = [[], []]
        loaded_labels = []
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
                        raise FileNotFoundError(
                            'Dataset not found: {} ({}) {}'.format(split, self.args.data, path))
                datasets.append(
                    TokenBlockDataset(
                        ds, ds.sizes, 0, pad=self.dictionary.pad(), eos=self.dictionary.eos(),
                        break_mode='eos', include_targets=False,
                    ))

            if stop:
                break

            raw_path = os.path.join(tokenized_data_path, split_k)
            loaded_labels = []
            with open(os.path.join(raw_path, 'lbl.txt'), 'r') as lbl_f:
                lines = lbl_f.readlines()
                for line in lines:
                    lbls = int(line.strip())
                    loaded_labels.append(lbls)

            print('| {} {} {} examples'.format(
                self.args.data, split_k, len(loaded_datasets[0][-1])))

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

        self.datasets[split] = ParagraphRankingDataset(
            dataset_q, dataset_c, loaded_labels, sizes_q + sizes_c, self.dictionary, True if split == 'train' else False, self.max_length
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
            assert self.num_labels == 2
            probs = logging_output['lprobs'].exp()
            pos = sample['target'].view(-1).eq(1)
            neg = sample['target'].view(-1).eq(0)

            correct_pos = probs[pos] > 1.0 / self.num_labels
            correct_neg = probs[neg] > 1.0 / self.num_labels

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

            # if False:
            #     correct = pos.new_zeros(pos.shape)
            #     correct[pos] = correct_pos
            #     correct[neg] = correct_neg
            #     incorrect = ~correct
            #     incorrect_ids = sample['id'][incorrect.nonzero()]
            #     logging_output['extra_metrics']['misclassified'] = incorrect_ids.squeeze().tolist()

        return loss, sample_size, logging_output

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary
