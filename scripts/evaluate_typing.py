from collections import Counter
import sys
import torch
import os

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

def score(key, prediction):


def collate(samples):

    pad_idx = PADIDX
    num_class = 113

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
    def __init__(self, task, data_path, max_length, use_marker):
        super().__init__()
        self.task = task
        self.data_path = data_path
        self.raw_data = self.load_dataset()
        self.vocab = task.dictionary
        self.max_length = max_length
        self.use_marker = use_marker

    def __getitem__(self, index):
        raw_sample = self.raw_data[index]
        sent = raw_sample['sent']
        id_ = raw_sample['id']
        rel_label = raw_sample["lbl"]
        e1_offset_orig = raw_sample['e1_start']
        e2_offset_orig = raw_sample['e2_start']
        e1_len = raw_sample['e1_len']
        e2_len = raw_sample['e2_len']

        e1_end = e1_offset_orig + e1_len
        e2_end = e2_offset_orig + e2_len

        block_text = torch.LongTensor(self.binarize_list(sent))
        block_text = block_text.tolist()

        if self.use_marker:
            e1_start_marker = self.vocab.index("[unused1]")
            e1_end_marker = self.vocab.index("[unused2]")
            e2_start_marker = self.vocab.index("[unused3]")
            e2_end_marker = self.vocab.index("[unused4]")
            if e1_offset_orig < e2_offset_orig:
                assert e1_end <= e2_offset_orig
                block_text = block_text[:e1_offset_orig] + \
                    [e1_start_marker] + \
                    block_text[e1_offset_orig:e1_end] + \
                    [e1_end_marker] +  \
                    block_text[e1_end:e2_offset_orig] + \
                    [e2_start_marker] + \
                    block_text[e2_offset_orig:e2_end] + \
                    [e2_end_marker] + \
                    block_text[e2_end:]

                e1_offset = 1 + e1_offset_orig
                e2_offset = 3 + e2_offset_orig

            else:
                assert e2_end <= e1_offset_orig
                block_text = block_text[:e2_offset_orig] + \
                    [e2_start_marker] + \
                    block_text[e2_offset_orig:e2_end] + \
                    [e2_end_marker] + \
                    block_text[e2_end:e1_offset_orig] + \
                    [e1_start_marker] + \
                    block_text[e1_offset_orig:e1_end] + \
                    [e1_end_marker] + \
                    block_text[e1_end:]
                e2_offset = 1 + e2_offset_orig
                e1_offset = 3 + e1_offset_orig

        block_text = torch.LongTensor(block_text)
        text, segment = self.prepend_cls(block_text)

        if self.use_marker:
            assert self.debinarize_list(text.tolist())[
                e1_offset] == '[unused1]'
            assert self.debinarize_list(text.tolist())[
                e2_offset] == '[unused3]'

        # truncate the sample
        item_len = text.size(0)
        if item_len > self.max_length:
            text = text[:self.max_length]
            segment = segment[:self.max_length]
            e1_offset = min(e1_offset, self.max_length - 1)
            e2_offset = min(e2_offset, self.max_length - 1)

        return {'id': id_, "segment": segment, "text": text, "target": rel_label, "e1_offset": e1_offset, "e2_offset": e2_offset}

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
        raw_path = self.raw_path
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
            

        return samples

if __name__ == '__main__':
    parser = options.get_training_parser('re')
    parser.add_argument('--model-path', default='/checkpoint/xwhan/2019-08-12/re_marker_only_bert_large.re.adam.lr1e-05.bert_large.crs_ent.seed3.bsz4.maxlen256.drop0.2.ngpu8/checkpoint_best.pt')
    parser.add_argument('--eval-data', default='/private/home/xwhan/dataset/tacred/processed-splits/valid', type=str)
    parser.add_argument('--eval-bsz', default=64, type=int)
    args = options.parse_args_and_arch(parser)

    task = tasks.setup_task(args)
    model = _load_models(args, task)
    model.cuda()

    eval_dataset = REDataset(task, args.eval_data, args.max_length, use_marker=True)
    relation2id = eval_dataset.relation2id
    id2relation = {v: k for k, v in relation2id.items()}
    dataloader = DataLoader(eval_dataset, batch_size=args.eval_bsz, collate_fn=collate, num_workers=20)

    id2gold = {}
    id2pred = {}

    with torch.no_grad():
        for batch_ndx, batch_data in enumerate(tqdm(dataloader)):
            batch_cuda = move_to_cuda(batch_data)
            logits = model(**batch_cuda['net_input'])
            pred = torch.argmax(logits, dim=-1).tolist()
            gold = batch_data["target"].squeeze().tolist()
            ids = batch_data["ids"]
            for id_, p, g in zip(ids, pred, gold):
                id2pred[id_] = id2relation[p]
                id2gold[id_] = id2relation[g]
    score(id2gold, id2pred)

