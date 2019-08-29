"""
use trained ranker to rank all DrQA paragraphs and use the topk for final answer evaluation
"""

from multiprocessing import Pool
import json
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict

sys.path.append('/private/home/xwhan/fairseq-py')

from fairseq import checkpoint_utils, tasks, options
from fairseq.utils import move_to_cuda
from fairseq.data import data_utils
from torch.utils.data import DataLoader, Dataset
from fairseq.tokenization import BertTokenizer
import numpy as np
import hashlib

from evaluate_reader import build_map

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

def _process_samples(items, tokenizer):
    all_samples = []
    for item in items:
        labels = item['para_has_answer']
        orig_scores = item['scores']
        question = item['question']
        paras = item['para']
        qid = hash_q_id(item['question'])
        q_toks = tokenizer.tokenize(question)
        for para_idx, (para, lbl, score) in enumerate(zip(paras, labels, orig_scores)):
            para_toks = tokenizer.tokenize(para)
            all_samples.append({'qid': qid, 'q_toks': q_toks, 'para_toks': para_toks, 'label': int(lbl), 'para_id': para_idx, 'orig_score': score, 'q': question, 'para': para})
    return all_samples


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
        q = raw_sample['q_toks']
        qid = raw_sample['qid']
        para = raw_sample['para_toks']
        lbl = raw_sample.get('label', -1)
        orig_score = raw_sample['orig_score']

        ques_idx = torch.LongTensor(self.binarize_list(q))
        para_idx = torch.LongTensor(self.binarize_list(para))
        if para_idx.size(0) + ques_idx.size(0) + 3 > 512:
            extra_len = para_idx.size(0) + ques_idx.size(0) + 3 - 512
            para_idx = para_idx[:-extra_len]
        sent1 = torch.cat([ques_idx.new(1).fill_(self.vocab.cls()), ques_idx, ques_idx.new(1).fill_(self.vocab.sep())])
        seg1 = torch.zeros(sent1.size(0)).long()
        sent2 = torch.cat([para_idx, para_idx.new(1).fill_(self.vocab.sep())])
        seg2 = torch.ones(sent2.size(0)).long()
        seg = torch.cat([seg1, seg2])
        sent = torch.cat([sent1, sent2])

        return {'qid': qid, 'para_id': raw_sample['para_id'], 'sentence':sent, 'segment': seg, 'target': torch.LongTensor([lbl]), 'orig_score': orig_score}

    def __len__(self):
        return len(self.raw_data)
    
    def binarize_list(self, words):
        return [self.vocab.index(w) for w in words]

    def load_dataset(self):
        """
        load test paragraphs and tokenize questions and paragraphs
        """
        tokenizer = BertTokenizer(
        '/private/home/xwhan/fairseq-py/vocab_dicts/vocab.txt', do_lower_case=True)
        data = [json.loads(item) for item in open(self.data_path)]

        num_workers = 30
        chunk_size= len(data) // num_workers
        offsets= [_ * chunk_size for _ in range(0, num_workers)] + [len(data)]
        pool= Pool(processes=num_workers)
        print(f'Start multi-processing with {num_workers} workers....')
        results= [pool.apply_async(_process_samples, args=(data[offsets[work_id]: offsets[work_id + 1]], tokenizer)) for work_id in range(num_workers)]
        outputs= [p.get() for p in results]
        samples= []
        for o in outputs:
            samples.extend(o)
        
        print(f'Load {len(samples)} paragraphs in total')

        return samples

def collate(samples):
    if len(samples) == 0:
        return {}
    return {
        'id': [s['qid'] for s in samples],
        'para_id': [s['para_id'] for s in samples],
        'orig_score': [s['orig_score'] for s in samples],
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
    
    model.make_generation_fast_()
    model.half()
    model.eval()
    model.cuda()

    model = nn.DataParallel(model)

    eval_dataset = RankerDataset(task, args.eval_data)
    dataloader = DataLoader(eval_dataset, batch_size=args.eval_bsz,
                            collate_fn=collate, num_workers=20, pin_memory=True)

    qid_score_label = []
    qid_paraid_scores = defaultdict(list)

    with torch.no_grad():
        for batch_ndx, batch_data in enumerate(tqdm(dataloader)):
            batch_cuda = move_to_cuda(batch_data)
            out = model(**batch_cuda['net_input'])
            logits = F.softmax(out, dim=-1)
            scores = logits[:,1].tolist()
            labels = batch_data['target'].view(-1).tolist()
            orig_scores = batch_data['orig_score']
            for qid, para_id, score in zip(batch_data['id'], batch_data['para_id'], scores):
                qid_paraid_scores[qid].append((para_id, score))

            qid_score_label += list(zip(batch_data['id'], scores, labels, orig_scores))


    qid2results = defaultdict(list)
    qid2orig_results = defaultdict(list)
    for _ in qid_score_label:
        qid = _[0]
        qid2results[qid].append((_[1],_[2]))
        qid2orig_results[qid].append((_[3], _[2]))
    
    print(f'BERT reranking results:')
    hits(qid2results)

    print(f'DrQA ranking results:')
    hits(qid2orig_results)

    topk_paras = defaultdict(dict)
    for k, v in qid_paraid_scores.items():
        v = sorted(v, key=lambda x:x[1], reverse=True)
        for para_id, score in v:
            topk_paras[k][para_id] = score

    tokenizer = BertTokenizer(
        '/private/home/xwhan/fairseq-py/vocab_dicts/vocab.txt', do_lower_case=True)

    # prepare data for qa
    with open(args.save_path, 'w') as f:
        # write scores to raw data for finding answers 
        num_workers = 30
        chunk_size = len(eval_dataset.raw_data) // num_workers
        offsets = [
            _ * chunk_size for _ in range(0, num_workers)] + [len(eval_dataset.raw_data)]
        pool = Pool(processes=num_workers)
        print(f'Start multi-processing with {num_workers} workers....')
        results = [pool.apply_async(_process_qa_samples, args=(
            eval_dataset.raw_data[offsets[work_id]: offsets[work_id + 1]], tokenizer, topk_paras)) for work_id in range(num_workers)]
        outputs = [p.get() for p in results]
        samples = []
        for o in outputs:
            samples.extend(o)
        for s in samples:
            f.write(json.dumps(s) + '\n')
        print(f'Wrote {len(samples)} paragraphs in total')


def _process_qa_samples(samples, tokenizer, topk_paras):
    outputs = []
    for sample in samples:
        qid = sample['qid']
        para_id = sample['para_id']
        sample['q_subtoks'] = sample['q_toks']
        sample['para_subtoks'] = sample['para_toks']
        del sample['q_toks']
        del sample['para_toks']
        orig_to_tok_index, tok_to_orig_index, doc_tokens, wp_tokens = build_map(sample['para'], tokenizer)
        sample['tok_to_orig_index'] = tok_to_orig_index
        sample['para_toks'] = doc_tokens
        if para_id in topk_paras[qid]:
            sample['score'] = topk_paras[qid][para_id]
            outputs.append(sample)
    return outputs

if __name__ == '__main__':
    parser = options.get_training_parser('span_qa')
    # parser.add_argument('--criterion', default='cross_entropy')
    parser.add_argument('--model-path', metavar='FILE', help='path(s) to model file(s), colon separated',
                        default='/checkpoint/xwhan/2019-08-18/WebQ_ranking_baseline.finetuning_paragraph_ranker.adam.lr1e-05.bert.crs_ent.seed3.bsz8.ldrop0.2.ngpu1/checkpoint_best.pt')
    parser.add_argument('--topk', default=20, type=int, help="how many paragraphs selected for open-domain QA")
    parser.add_argument('--eval-data', default='/private/home/xwhan/dataset/triviaqa/raw/valid.json', type=str)
    parser.add_argument('--save-path', default='/private/home/xwhan/dataset/triviaqa/raw/valid_with_scores_best.json', type=str)
    parser.add_argument('--eval-bsz', default=128, type=int)

    args = options.parse_args_and_arch(parser)
    # args = parser.parse_args()

    main(args)
