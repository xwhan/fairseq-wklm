from collections import Counter
import sys
import torch
import os
from functools import partial
import torch.nn as nn

sys.path.append('/private/home/xwhan/fairseq-py')
from fairseq import checkpoint_utils, tasks, options
from torch.utils.data import DataLoader, Dataset
from fairseq.data import data_utils
from fairseq.utils import move_to_cuda
from tqdm import tqdm
import json


def _load_models(args, task):
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.model_path.split(':'), arg_overrides=vars(args),
        task=task,
    )
    for model in models:
        model.eval()
    return models[0]

def collate(samples, pad_idx=0, num_class=113):
    if len(samples) == 0:
        return {}
    if not isinstance(samples[0], dict):
        samples = [s for sample in samples for s in sample]

    batch_text = data_utils.collate_tokens(
        [s['text'] for s in samples], pad_idx, left_pad=False)

    masks = torch.zeros(batch_text.size(0), batch_text.size(1))
    target = torch.zeros(batch_text.size(0), num_class)

    for idx, s in enumerate(samples):
        e_offset = s['e_offset']
        masks[idx, e_offset] = 1
        for t in s['target']:
            target[idx, t] = 1

    return {
        'ntokens': sum(len(s['text']) for s in samples),
        'net_input': {
            'sentence': batch_text,
            'segment': data_utils.collate_tokens(
                [s['segment'] for s in samples], pad_idx, left_pad=False,
            ),
            'entity_masks': masks
        },
        'target': target,
        'nsentences': len(samples),
    }

class TypeDataset(Dataset):
    """docstring for TypeDataset"""
    def __init__(self, task, data_path, max_length, use_marker, use_sep):
        super().__init__()
        self.task = task
        self.data_path = data_path
        self.raw_data = self.load_dataset()
        self.vocab = task.dictionary
        self.max_length = max_length
        self.use_marker = use_marker
        self.use_sep = use_sep

    def __getitem__(self, index):
        raw_sample = self.raw_data[index]
        sent = raw_sample['sent']
        type_label = raw_sample["lbl"]
        e_offset_orig = raw_sample['e_start']
        e_len = raw_sample['e_len']

        e_end = e_offset_orig + e_len
        block_text = self.binarize_list(sent)

        if self.use_marker:
            e_start_marker = self.vocab.index("[unused1]")
            e_end_marker = self.vocab.index("[unused2]")

            block_text = block_text[:e_offset_orig] + [e_start_marker] + block_text[e_offset_orig:e_end] + \
                [e_end_marker] + block_text[e_end:] + \
                [self.vocab.index("[SEP]")]
            e_offset = 1 + e_offset_orig

            block_text = torch.LongTensor(block_text)
            sent, segment = self.prepend_cls(block_text)

        elif self.use_sep:
            orig_sent_len = len(block_text)
            block_text = block_text + \
                [self.vocab.index(
                    "[SEP]")] + block_text[e_offset_orig:e_end] + [self.vocab.index("[SEP]")]
            e_offset = 1 + e_offset_orig
            block_text = torch.LongTensor(block_text)
            sent, segment = self.prepend_cls(block_text)
            segment[orig_sent_len + 2:] = 1

        if self.use_marker:
            assert self.debinarize_list(sent.tolist())[
                e_offset] == '[unused1]'

        # truncate the sample
        item_len = sent.size(0)
        if item_len > self.max_length:
            sent = sent[:self.max_length]
            segment = segment[:self.max_length]
            e_offset = min(e_offset, self.max_length - 1)
    
        return {'text': sent, 'segment': segment, 'target': type_label, 'e_offset': e_offset}

    def prepend_cls(self, sent):
        cls = sent.new_full((1,), self.vocab.cls())
        sent = torch.cat([cls, sent])
        segment = torch.zeros(sent.size(0)).long()
        return sent, segment

    def binarize_list(self, words):
        """
        binarize tokenized sequence
        """
        return [self.vocab.index(w) for w in words]

    def debinarize_list(self, indice):
        return [self.vocab[int(idx)] for idx in indice]

    def tokenize(self, s):
        try:
            return self.task.tokenizer.tokenize(s)
        except:
            print('failed on', s)
            raise

    def __len__(self):
        return len(self.raw_data)

    def load_dataset(self):
        """
        load datasets from tokenized 
        """
        raw_path = self.data_path
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

        sents = []
        with open(os.path.join(raw_path, 'sent.txt'), 'r') as lbl_f:
            lines = lbl_f.readlines()
            for line in lines:
                toks = line.strip().split()
                sents.append(toks)

        samples = []
        for sent, e_offset, e_len, lbl in zip(sents, e_offsets, e_lens, loaded_labels):
            samples.append({'sent': sent, 'e_start':e_offset, 'e_len': e_len, 'lbl': lbl})
        
        return samples

def score(strict_acc, n_pred, n_true, n_corr, ma_p, ma_r, eval_size):

    def f1(p, r):
        if p == 0 or r == 0:
            return 0.0
        else:
            return 2.0 * p * r / (p + r)

    strict_acc = strict_acc / eval_size

    if n_pred > 0:
        mi_p = n_corr / n_pred
    else:
        mi_p = 0
    mi_r = n_corr /n_true
    mi_f1 = f1(mi_p, mi_r)

    ma_p = ma_p / eval_size
    ma_r = ma_r / eval_size
    ma_f1 = f1(ma_p, ma_r)

    metrics = {'acc': strict_acc, 'mi_f1': mi_f1, 'ma_f1': ma_f1}
    print(metrics)
    return metrics

if __name__ == '__main__':
    parser = options.get_training_parser('typing')
    parser.add_argument('--model-path', default='/checkpoint/xwhan/2019-08-12/re_marker_only_bert_large.re.adam.lr1e-05.bert_large.crs_ent.seed3.bsz4.maxlen256.drop0.2.ngpu8/checkpoint_best.pt')
    parser.add_argument('--eval-data', default='/private/home/xwhan/dataset/FIGER/processed-splits/valid', type=str)
    parser.add_argument('--eval-bsz', default=128, type=int)
    args = options.parse_args_and_arch(parser)

    task = tasks.setup_task(args)
    model = _load_models(args, task)
    model.half()
    model.eval()
    model.cuda()
    model = nn.DataParallel(model)

    eval_dataset = TypeDataset(task, args.eval_data, args.max_length,
                               use_marker=args.use_marker, use_sep=args.use_sep)
    collate_fn = partial(collate, pad_idx=task.dictionary.pad(), num_class=task.num_class)
    dataloader = DataLoader(eval_dataset, batch_size=args.eval_bsz, collate_fn=collate, num_workers=20)

    n_pred = n_true = n_corr = ma_p = ma_r = strict_acc = eval_size = 0
    with torch.no_grad():
        for batch_ndx, batch_data in enumerate(tqdm(dataloader)):
            batch_cuda = move_to_cuda(batch_data)
            net_output = model(**batch_cuda['net_input'])
            type_probs = torch.sigmoid(net_output)
            targets = batch_data['target']
            assert type_probs.size(0) == targets.size(0)
            sample_size = type_probs.size(0)
            eval_size += sample_size
            for idx in range(sample_size):
                pred_probs = type_probs[idx, :].tolist()
                target = targets[idx, :]
                true_labels, predicted_labels = [], []

                max_prob = max(pred_probs)
                for type_idx, prob in enumerate(pred_probs):
                    if prob > args.thresh or prob == max_prob:
                        predicted_labels.append(type_idx)
                    if target[type_idx] == 1:
                        true_labels.append(type_idx)

                strict_acc += set(predicted_labels) == set(true_labels)
                n_pred += len(predicted_labels)
                n_true += len(true_labels)
                n_corr += len(set(predicted_labels).intersection(set(true_labels)))

                ma_p += len(set(predicted_labels).intersection(
                    set(true_labels))) / float(len(predicted_labels))
                ma_r += len(set(predicted_labels).intersection(set(true_labels))
                            ) / float(len(true_labels))
    
    print(f'evaluating {eval_size} examples with thresh {args.thresh}')
    score(strict_acc, n_pred, n_true, n_corr, ma_p, ma_r, eval_size)
