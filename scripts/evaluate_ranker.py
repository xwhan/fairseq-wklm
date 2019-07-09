import json
import sys
import torch
from tqdm import tqdm
import torch.nn.functional as F

from collections import defaultdict

sys.path.append('/private/home/xwhan/fairseq-py')

from fairseq import checkpoint_utils, tasks, options
from fairseq.utils import move_to_cuda

from fairseq.data import data_utils

from torch.utils.data import DataLoader, Dataset
import numpy as np

import hashlib

def hash_q_id(question):
    return hashlib.md5(question.encode()).hexdigest()


def _load_models(args, task):
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.model_path.split(':'),
        task=task,
    )
    for model in models:
        model.eval()
    return models[0]


class RankerDataset(Dataset):
    """docstring for RankerDataset"""
    def __init__(self, task, data_path):
        super().__init__()
        self.task = task
        self.data_path = data_path
        self.raw_data = self.load_dataset()
        self.vocab = task.dictionary

    def __getitem__(self, index):
        raw_sample = self.raw_data[index]
        question = raw_sample['q']
        qid = raw_sample['qid']
        para = raw_sample['para']
        lbl = raw_sample['label']
        ques_idx = self.task.dictionary.encode_line(question.lower(), line_tokenizer=self.task.tokenizer.tokenize, append_eos=False)
        para_idx = self.task.dictionary.encode_line(para.lower(), line_tokenizer=self.task.tokenizer.tokenize, append_eos=False)
        if len(para_idx.tolist() + ques_idx.tolist()) + 3 > 512:
            extra_len = len(para_idx.tolist() + ques_idx.tolist()) + 3 - 512
            para_idx = para_idx[:-extra_len]
        sent1 = torch.cat([ques_idx.new(1).fill_(self.vocab.cls()), ques_idx, ques_idx.new(1).fill_(self.vocab.sep())])
        seg1 = torch.zeros(sent1.size(0)).long()
        sent2 = torch.cat([para_idx, para_idx.new(1).fill_(self.vocab.sep())])
        seg2 = torch.ones(sent2.size(0)).long()
        seg = torch.cat([seg1, seg2])
        sent = torch.cat([sent1, sent2])

        return {'qid': qid, 'para_id': raw_sample['para_id'], 'sentence':sent, 'segment': seg, 'target': torch.LongTensor([lbl])}

    def __len__(self):
        return len(self.raw_data)

    def load_dataset(self):
        raw_data = [json.loads(item) for item in open(self.data_path).readlines()]
        samples = []
        for item in raw_data:
            labels = item['para_has_answer']
            question = item['question']
            qid = hash_q_id(question)
            paras = item['para']
            for para_idx, (para, lbl) in enumerate(zip(paras, labels)):
                samples.append({'qid': qid, 'q':question, 'para': para, 'label': int(lbl), 'para_id': para_idx})
        return samples

def collate(samples):
    if len(samples) == 0:
        return {}

    return {
        'id': [s['qid'] for s in samples],
        'para_id': [s['para_id'] for s in samples],
        'ntokens': sum(len(s['sentence']) for s in samples),
        'net_input': {
            'sentence': data_utils.collate_tokens(
                [s['sentence'] for s in samples], 0, 100, left_pad=False,
            ),
            'segment_labels': data_utils.collate_tokens(
                [s['segment'] for s in samples], 0, 100, left_pad=False,
            ),
        },
        'target': torch.stack([s['target'] for s in samples], dim=0),
        'nsentences': samples[0]['sentence'].size(0),
    }


def hits(qid2results, k=[1,5,10,20]):
    for k_ in k:
        hitsk = []
        for qid in qid2results.keys():
            score_label = qid2results[qid]
            ranked = sorted(score_label, key=lambda x:x[0], reverse=True)
            ranked = ranked[:k_]
            top_labels = [_[1] for _ in ranked]
            hitsk.append(int(np.sum(top_labels) > 0))
        print(f'hits at {k_} is {np.mean(hitsk)} for {len(qid2results.keys())} questions')


def main(args):
    task = tasks.setup_task(args)

    model = _load_models(args, task)

    model.cuda()
    model.eval()

    eval_dataset = RankerDataset(task, args.eval_data)
    dataloader = DataLoader(eval_dataset, batch_size=args.eval_bsz, collate_fn=collate, num_workers=40)

    qid_score_label = []
    qid_paraid_scores = defaultdict(dict)

    with torch.no_grad():
        for batch_ndx, batch_data in enumerate(tqdm(dataloader)):
            batch_cuda = move_to_cuda(batch_data)
            out = model(**batch_cuda['net_input'])
            logits = F.softmax(out, dim=-1)
            scores = logits[:,1].tolist()
            labels = batch_data['target'].view(-1).tolist()
            for qid, para_id, score in zip(batch_data['id'], batch_data['para_id'], scores):
                qid_paraid_scores[qid][para_id] = score

            qid_score_label += list(zip(batch_data['id'], scores, labels))

    qid2results = defaultdict(list)
    for _ in qid_score_label:
        qid = _[0]
        qid2results[qid].append((_[1],_[2]))

    hits(qid2results)

    with open(args.save_path, 'w') as f:
        # write scores to raw data for QA
        for sample in eval_dataset.raw_data:
            qid = sample['qid']
            para_id = sample['para_id']
            sample['score'] = qid_paraid_scores[qid][para_id]
            f.write(json.dumps(sample) + '\n')

if __name__ == '__main__':
    parser = options.get_parser('Trainer', 'paragaph_ranking')
    options.add_dataset_args(parser)
    parser.add_argument('--criterion', default='cross_entropy')
    parser.add_argument('--model-path', metavar='FILE', help='path(s) to model file(s), colon separated', default='/checkpoint/xwhan/2019-07-05/ranker_neg10.finetuning_paragraph_ranker.mxup100000.adam.lr1e-05.bert.crs_ent.seed3.bsz8.ldrop0.2.ngpu1/checkpoint_best.pt')
    parser.add_argument('--eval-data', default='/private/home/xwhan/dataset/webq_ranking/webq_test.json', type=str)
    parser.add_argument('--save-path', default='/private/home/xwhan/dataset/webq_ranking/webq_test_with_scores.json', type=str)
    parser.add_argument('--eval-bsz', default=64, type=int)

    args = options.parse_args_and_arch(parser)
    args = parser.parse_args()

    main(args)
