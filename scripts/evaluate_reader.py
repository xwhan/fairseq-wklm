from multiprocessing import Pool
import hashlib
import json
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import os

from collections import defaultdict

sys.path.append('/private/home/xwhan/fairseq-py')

from fairseq import checkpoint_utils, tasks, options
from fairseq.utils import move_to_cuda

from fairseq.data import data_utils

from torch.utils.data import DataLoader, Dataset
import numpy as np

import collections
from fairseq.tokenization import BasicTokenizer

from official_eval import f1_score, exact_match_score, metric_max_over_ground_truths


def _load_models(args, task):
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.model_path.split(':'), {"last_dropout": 0}, task=task,
    )
    for model in models:
        model.eval()
    return models[0]

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def get_final_text(pred_text, orig_text, tokenizer, do_lower_case, verbose_logging=False):

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            print(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            print("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            print("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            print("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text

def build_map(context, tokenizer):
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in context:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    orig_to_tok_index = [] # original token to wordpiece index
    tok_to_orig_index = [] # wordpiece token to original token index
    all_doc_tokens = [] # wordpiece tokens
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token) # wordpiece tokens
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    return orig_to_tok_index, tok_to_orig_index, doc_tokens, all_doc_tokens


def process_raw_items(items, tokenizer):
    samples = []
    for item in items:
        sample = {}
        question_toks = item['q_toks']
        orig_to_tok_index, tok_to_orig_index, doc_tokens, wp_tokens = build_map(item['para'], tokenizer)
        sample['tok_to_orig_index'] = tok_to_orig_index
        sample['para_subtoks'] = wp_tokens
        sample['para_toks'] = doc_tokens
        sample['q_subtoks'] = question_toks
        sample['qid'] = item['qid']
        sample['score'] = item.get('score', 0)
        sample['para_id'] = item['para_id']
        sample['q'] = item['q']
        sample['para'] = item['para']
        samples.append(sample)
    return samples


def process_raw(data_path, tokenizer, out_path):
    raw_data = [json.loads(item) for item in open(data_path).readlines()]  

    num_workers = 30
    chunk_size = len(raw_data) // num_workers
    offsets = [
        _ * chunk_size for _ in range(0, num_workers)] + [len(raw_data)]
    pool = Pool(processes=num_workers)
    print(f'Start multi-processing with {num_workers} workers....')
    results = [pool.apply_async(process_raw_items, args=(
        raw_data[offsets[work_id]: offsets[work_id + 1]], tokenizer)) for work_id in range(num_workers)]
    outputs = [p.get() for p in results]
    samples = []
    for o in outputs:
        samples.extend(o)
    # for item in tqdm(raw_data):
    #     q = item['q']
    #     para = item['para']
    #     question_toks = tokenizer.tokenize(q)
    #     orig_to_tok_index, tok_to_orig_index, doc_tokens, wp_tokens = build_map(para, tokenizer)
    #     item['tok_to_orig_index'] = tok_to_orig_index
    #     item['para_subtoks'] = wp_tokens
    #     item['para_toks'] = doc_tokens
    #     item['q_subtoks'] = question_toks

    with open(out_path, 'w') as g:
        for _ in samples:
            g.write(json.dumps(_) + '\n')


def hash_q_id(question):
    return hashlib.md5(question.encode()).hexdigest()

class ReaderDataset(Dataset):
    """docstring for RankerDataset"""
    def __init__(self, task, data_path, max_query_lengths, max_length, downsample=1.0):
        super().__init__()
        self.task = task
        self.data_path = data_path
        self.raw_data = self.load_dataset()

        self.raw_data = self.raw_data[:int(downsample * len(self.raw_data))]
        self.vocab = task.dictionary
        self.max_query_length = max_query_lengths
        self.max_length = max_length

    def __getitem__(self, index):
        raw_sample = self.raw_data[index]
        qid = raw_sample['qid']
        para_id = raw_sample['para_id']
        score = raw_sample.get('score', 0)

        q_subtoks = raw_sample['q_subtoks'] 
        question = torch.LongTensor(self.binarize_list(q_subtoks))
        para_subtoks = raw_sample['para_subtoks']
        paragraph = torch.LongTensor(self.binarize_list(para_subtoks))

        if question.size(0) > self.max_query_length:
            question = question[:self.max_query_length]
        para_offset = question.size(0) + 2
        max_tokens_for_doc = self.max_length - para_offset - 1
        if paragraph.size(0) > max_tokens_for_doc:
            paragraph = paragraph[:max_tokens_for_doc]
        text, seg = self._join_sents(question, paragraph)
        paragraph_mask = torch.zeros(text.shape).byte()
        paragraph_mask[para_offset:-1] = 1

        return {'qid': qid, 'para_id': para_id, 'sentence':text, 'segment': seg, 'score': score, 'para_offset': para_offset, 'paragraph_mask': paragraph_mask, 'doc_tokens': raw_sample['para_toks'], 'wp_tokens': para_subtoks, 'tok_to_orig_index': raw_sample['tok_to_orig_index'], 'q': raw_sample['q'], 'c': raw_sample['para']}

    def _join_sents(self, sent1, sent2):
        cls = sent1.new_full((1,), self.vocab.cls())
        sep = sent1.new_full((1,), self.vocab.sep())
        sent1 = torch.cat([cls, sent1, sep])
        sent2 = torch.cat([sent2, sep])
        text = torch.cat([sent1, sent2])
        segment1 = torch.zeros(sent1.size(0)).long()
        segment2 = torch.ones(sent2.size(0)).long()
        segment = torch.cat([segment1, segment2])
        return text, segment

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
        print('Loading eval data...')
        raw_data = [json.loads(item.strip()) for item in open(self.data_path).readlines()]
        samples = []
        for item in raw_data:
            item["score"] = item.get('score', 1.0)
            samples.append(item)
        return samples

def collate(samples):
    if len(samples) == 0:
        return {}

    return {
        'id': [s['qid'] for s in samples],
        'q': [s['q'] for s in samples],
        'c': [s['c'] for s in samples],
        'para_id': [s['para_id'] for s in samples],
        'doc_tokens': [s['doc_tokens'] for s in samples],
        'wp_tokens': [s['wp_tokens'] for s in samples],
        'ntokens': sum(len(s['sentence']) for s in samples),
        'tok_to_orig_index': [s['tok_to_orig_index'] for s in samples],
        'scores': [s['score'] for s in samples],
        'para_offset': [s['para_offset'] for s in samples],
        'net_input': {
            'text': data_utils.collate_tokens(
                [s['sentence'] for s in samples], 0, 100, left_pad=False,
            ),
            'segment': data_utils.collate_tokens(
                [s['segment'] for s in samples], 0, 100, left_pad=False,
            ),
            'paragraph_mask':data_utils.collate_tokens(
                [s['paragraph_mask'] for s in samples], 0, 100, left_pad=False,
            ),

        },
        'nsentences': samples[0]['sentence'].size(0),
    }


def main(args):
    task = tasks.setup_task(args)

    # process_raw("/private/home/xwhan/dataset/squad1.1/splits/valid.json", task.tokenizer, "/private/home/xwhan/dataset/squad1.1/splits/valid_eval.json")
    # assert False

    # process_raw("/private/home/xwhan/dataset/triviaqa/raw/valid_with_scores_best.json", task.tokenizer,
    #             "/private/home/xwhan/dataset/triviaqa/raw/valid_eval.json")
    # assert False

    model = _load_models(args, task)
    model.half()
    model.eval()
    model.cuda()
    model = nn.DataParallel(model)

    eval_dataset = ReaderDataset(task, args.eval_data, args.max_query_length, args.max_length, args.downsample)
    dataloader = DataLoader(eval_dataset, batch_size=args.eval_bsz, collate_fn=collate, num_workers=20)

    qid2results = defaultdict(list)
    qid2question = {} # for SQuAD analysis

    basic_tokenizer = BasicTokenizer(do_lower_case=True)

    print('Starting evaluation...')
    with torch.no_grad():
        for batch_ndx, batch_data in enumerate(tqdm(dataloader)):
            batch_cuda = move_to_cuda(batch_data)
            start_out, end_out, paragraph_mask = model(**batch_cuda['net_input'])
            outs = (start_out, end_out)
            questions_mask = paragraph_mask.ne(1)
            paragraph_outs = [
                o.view(-1, o.size(1)).float().masked_fill(questions_mask,  -1e10).type_as(o) for o in outs]
            outs = paragraph_outs
            ranking_scores = batch_data['scores']
            para_offset = batch_data['para_offset']
            span_scores = outs[0][:,:,None] + outs[1][:,None]

            # only select spans with start<=end and the answer lengths
            max_answer_lens = 20
            max_seq_len = span_scores.size(1)
            span_mask = np.tril(np.triu(np.ones((max_seq_len, max_seq_len)), 0), max_answer_lens)
            span_mask = span_scores.data.new(max_seq_len, max_seq_len).copy_(torch.from_numpy(span_mask))
            span_scores = span_scores.float()
            span_mask = span_mask.float()
            span_scores = span_scores - 1e10 * (1 - span_mask[None].expand_as(span_scores))

            start_position = span_scores.max(dim=2)[0].max(dim=1)[1].tolist()
            end_position = span_scores.max(dim=1)[0].max(dim=1)[1].tolist()
            answer_scores = span_scores.max(dim=1)[0].max(dim=1)[0].tolist()

            start_position_ = list(np.array(start_position) - np.array(para_offset))
            end_position_ = list(np.array(end_position) - np.array(para_offset))
            
            for qid, doc_tokens, wp_tokens, tok_to_orig_index, start, end, ans_score, r_score, q, c, offset in zip(batch_data['id'], batch_data['doc_tokens'], batch_data['wp_tokens'], batch_data['tok_to_orig_index'], start_position_, end_position_, answer_scores, ranking_scores, batch_data['q'], batch_data['c'], para_offset):
                tok_tokens = wp_tokens[start:end+1]

                # in case of empty paragraph
                if len(tok_to_orig_index) == 0:
                    qid2results[qid].append(("", ans_score, r_score))
                    qid2question[qid] = {'q': q, 'c': c}
                    continue

                orig_doc_start = tok_to_orig_index[start]
                orig_doc_end = tok_to_orig_index[end]
                orig_tokens = doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)
                final_text = get_final_text(tok_text, orig_text, basic_tokenizer, True)

                qid2results[qid].append((final_text, ans_score, r_score))
                qid2question[qid] = {'q': q, 'c': c}

    # evaluation
    qid2pred = {}
    for qid in qid2results.keys():
        qid2results[qid].sort(key=lambda x:combine(x[1],x[2]), reverse=True)
        qid2pred[qid] = qid2results[qid][0][0]

    # load groundtruth
    qid2ground = {}
    analysis = {}
    with open(args.answer_path) as f:
        for line in f.readlines():
            line = json.loads(line)
            if 'qid' not in line:
                line['qid'] = hash_q_id(line['question'])
            qid2ground[line['qid']] = line['answer']
            analysis[line['qid']] = {"gold": line['answer'], "pred": qid2results[line['qid']][0], 'q': qid2question[line['qid']]['q'], 'c': qid2question[line['qid']]['c']}

    print(f'how many evaluation data: {len(qid2ground)}')
    
    # save the predictions for tuninng
    for alpha in list(np.arange(0, 0.200, 0.005)) + [1]:
        qid2pred = {}
        for qid in qid2results.keys():
            qid2results[qid].sort(key=lambda x: combine(x[1], x[2], alpha), reverse=True)
            qid2pred[qid] = qid2results[qid][0][0]
        f1_scores = [metric_max_over_ground_truths(
            f1_score, qid2pred[qid], qid2ground[qid]) for qid in qid2pred.keys()]
        em_scores = [metric_max_over_ground_truths(
            exact_match_score, qid2pred[qid], qid2ground[qid]) for qid in qid2pred.keys()]
        print(f'Alpha: {alpha}')
        print(f'f1 score {np.mean(f1_scores)}')
        print(f'em score {np.mean(em_scores)}')

    qids = list(qid2pred.keys())
    f1_scores = [metric_max_over_ground_truths(
        f1_score, qid2pred[qid], qid2ground[qid]) for qid in qids]
    em_scores = [metric_max_over_ground_truths(
        exact_match_score, qid2pred[qid], qid2ground[qid]) for qid in qids]

    for qid, f1, em in zip(qids, f1_scores, em_scores):
        analysis[qid]['f1'] = f1
        analysis[qid]['em'] = em

    if args.save:
        save_path = os.path.join(args.save_name)
        with open(save_path, 'w') as g:
            json.dump(analysis, g)

    print(f'f1 score {np.mean(f1_scores)}')
    print(f'em score {np.mean(em_scores)}')

def combine(s1, s2, alpha=0.1):
    """
    s1: answer score
    s2: retrieval score
    """
    return s1 * alpha + s2 * (1 - alpha)

def metrics(args):
    prediction = json.load(open('/private/home/xwhan/dataset/webq_qa/prediction.json'))

    # load groundtruth
    qid2ground = {}
    with open(args.answer_path) as f:
        for line in f.readlines():
            line = json.loads(line)
            qid2ground[line['qid']] = line['answer']

    qid2pred = {}
    for qid in prediction.keys():
        pred_list = prediction[qid]
        pred_list.sort(key=lambda x:combine(x[1], x[2]) , reverse=True)
        qid2pred[qid] = pred_list[0][0]

    f1_scores = [metric_max_over_ground_truths(f1_score, qid2pred[qid], qid2ground[qid]) for qid in qid2ground.keys()]

    em_scores = [metric_max_over_ground_truths(exact_match_score, qid2pred[qid], qid2ground[qid]) for qid in qid2ground.keys()]

    print(f'f1 score {np.mean(f1_scores)}')
    print(f'em score {np.mean(em_scores)}')   

if __name__ == '__main__':
    parser = options.get_training_parser('span_qa')
    parser.add_argument('--model-path', metavar='FILE', help='path(s) to model file(s), colon separated', default='/checkpoint/xwhan/2019-08-04/reader_ft.span_qa.mxup187500.adam.lr1e-05.bert.crs_ent.seed3.bsz8.ngpu1/checkpoint_best.pt')

    parser.add_argument(
        '--eval-data', default='/private/home/xwhan/dataset/WebQ/raw/test_with_scores_best.json', type=str)
    parser.add_argument(
        '--answer-path', default='/private/home/xwhan/dataset/WebQ/raw/test.json')
    parser.add_argument('--downsample', default=1.0, help='test on small portion of the data')

    # save the prediction file
    parser.add_argument('--save-name', default='predictions.json')
    parser.add_argument('--eval-bsz', default=256, type=int)
    parser.add_argument('--save', action='store_true')

    args = options.parse_args_and_arch(parser)

    main(args)

    # metrics(args)
