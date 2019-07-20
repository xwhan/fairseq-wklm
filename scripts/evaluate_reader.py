import json
import sys
import torch
from tqdm import tqdm

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
        args.model_path.split(':'),
        task=task,
    )
    for model in models:
        model.eval()
    return models[0]

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):

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

    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

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


class ReaderDataset(Dataset):
    """docstring for RankerDataset"""
    def __init__(self, task, data_path, max_query_lengths, max_length):
        super().__init__()
        self.task = task
        self.data_path = data_path
        self.raw_data = self.load_dataset()
        self.vocab = task.dictionary
        self.max_query_length = max_query_lengths
        self.max_length = max_length

    def __getitem__(self, index):
        raw_sample = self.raw_data[index]
        question = raw_sample['q']
        qid = raw_sample['qid']
        para = raw_sample['para'].lower()
        para_id = raw_sample['para_id']
        score = raw_sample['score']

        question = self.task.dictionary.encode_line(question.lower(), line_tokenizer=self.task.tokenizer.tokenize, append_eos=False)
        orig_to_tok_index, tok_to_orig_index, doc_tokens, wp_tokens = self.build_map(para)
        paragraph = torch.LongTensor(self.binarize_list(wp_tokens))

        if question.size(0) > self.max_query_length:
            question = question[:self.max_query_length]
        para_offset = question.size(0) + 2
        max_tokens_for_doc = self.max_length - para_offset - 1
        if paragraph.size(0) > max_tokens_for_doc:
            paragraph = paragraph[:max_tokens_for_doc]
        text, seg = self._join_sents(question, paragraph)
        paragraph_mask = torch.zeros(text.shape).byte()
        paragraph_mask[para_offset:-1] = 1

        return {'qid': qid, 'para_id': para_id, 'sentence':text, 'segment': seg, 'score': score, 'para_offset': para_offset, 'paragraph_mask': paragraph_mask, 'doc_tokens': doc_tokens, 'wp_tokens': wp_tokens, 'tok_to_orig_index': tok_to_orig_index, 'q': raw_sample['q'], 'c': raw_sample['para']}

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

    def build_map(self, context):
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
            sub_tokens = self.tokenize(token) # wordpiece tokens
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        return orig_to_tok_index, tok_to_orig_index, doc_tokens, all_doc_tokens

    def __len__(self):
        return len(self.raw_data)

    def load_dataset(self):
        raw_data = [json.loads(item) for item in open(self.data_path).readlines()]
        samples = []
        for item in raw_data:
            ranking_score = item.get('score', 1.0)
            question = item['q']
            para = item['para']
            qid = item['qid']
            para_id = item['para_id']
            samples.append({'qid': qid, 'q':question, 'para': para,  'para_id': para_id, 'score':ranking_score})
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

    model = _load_models(args, task)

    model.cuda()

    eval_dataset = ReaderDataset(task, args.eval_data, args.max_query_length, args.max_length)
    dataloader = DataLoader(eval_dataset, batch_size=args.eval_bsz, collate_fn=collate, num_workers=10)

    qid2results = defaultdict(list)

    start_preds = []
    end_preds = []

    with torch.no_grad():
        for batch_ndx, batch_data in enumerate(tqdm(dataloader)):
            batch_cuda = move_to_cuda(batch_data)
            start_out, end_out, paragraph_mask = model(**batch_cuda['net_input'])
            outs = (start_out, end_out)
            questions_mask = paragraph_mask.ne(1)
            paragraph_outs = [o.view(-1, o.size(1)).masked_fill(questions_mask, -1e10) for o in outs]
            outs = paragraph_outs
            ranking_scores = batch_data['scores']
            para_offset = batch_data['para_offset']
            span_scores = outs[0][:,:,None] + outs[1][:,None]

            # only select spans with start<=end and the answer lengths
            max_answer_lens = 20
            max_seq_len = span_scores.size(1)
            span_mask = np.tril(np.triu(np.ones((max_seq_len, max_seq_len)), 0), max_answer_lens)
            span_mask = span_scores.data.new(max_seq_len, max_seq_len).copy_(torch.from_numpy(span_mask))
            span_scores = span_scores - 1e10 * (1 - span_mask[None].expand_as(span_scores))

            start_position = span_scores.max(dim=2)[0].max(dim=1)[1].tolist()
            end_position = span_scores.max(dim=1)[0].max(dim=1)[1].tolist()
            answer_scores = span_scores.max(dim=1)[0].max(dim=1)[0].tolist()

            start_position_ = list(np.array(start_position) - np.array(para_offset))
            end_position_ = list(np.array(end_position) - np.array(para_offset))

            for qid, doc_tokens, wp_tokens, tok_to_orig_index, start, end, ans_score, r_score, q, c, offset in zip(batch_data['id'], batch_data['doc_tokens'], batch_data['wp_tokens'], batch_data['tok_to_orig_index'], start_position_, end_position_, answer_scores, ranking_scores, batch_data['q'], batch_data['c'], para_offset):

                tok_tokens = wp_tokens[start:end+1]
                orig_doc_start = tok_to_orig_index[start]
                orig_doc_end = tok_to_orig_index[end]
                orig_tokens = doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)
                final_text = get_final_text(tok_text, orig_text, True)

                qid2results[qid].append((final_text, ans_score, r_score))

                start_preds.append(start)
                end_preds.append(end)

    if args.save:
        with open(args.save_path, 'w') as g:
            json.dump(qid2results, g)


    # evaluation
    qid2pred = {}
    for qid in qid2results.keys():
        qid2results[qid].sort(key=lambda x:combine(x[1],x[2]), reverse=True)
        qid2pred[qid] = qid2results[qid][0][0]

    # load groundtruth
    qid2ground = {}
    with open(args.answer_path) as f:
        for line in f.readlines():
            line = json.loads(line)
            qid2ground[line['qid']] = line['answer']

    assert len(qid2ground) == len(qid2pred)

    f1_scores = [metric_max_over_ground_truths(f1_score, qid2pred[qid], qid2ground[qid]) for qid in qid2ground.keys()]

    em_scores = [metric_max_over_ground_truths(exact_match_score, qid2pred[qid], qid2ground[qid]) for qid in qid2ground.keys()]

    print(f'f1 score {np.mean(f1_scores)}')
    print(f'em score {np.mean(em_scores)}')

def combine(s1, s2, alpha=0.1):
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
    parser = options.get_parser('Trainer', 'span_qa')
    options.add_dataset_args(parser)
    parser.add_argument('--criterion', default='span_qa')
    parser.add_argument('--model-path', metavar='FILE', help='path(s) to model file(s), colon separated', default='/checkpoint/xwhan/2019-07-12/reader_ft.span_qa.mxup187500.adam.lr1e-05.bert.crs_ent.seed3.bsz8.ngpu1/checkpoint_last.pt')
    parser.add_argument('--eval-data', default='/private/home/xwhan/dataset/webq_ranking/webq_test_with_scores.json', type=str)
    parser.add_argument('--answer-path', default='/private/home/xwhan/dataset/webq_qa/splits/test.json')
    parser.add_argument('--save-path', default='/private/home/xwhan/dataset/webq_qa/prediction_ft.json')
    parser.add_argument('--eval-bsz', default=16, type=int)
    parser.add_argument('--save', action='store_true')
    args = options.parse_args_and_arch(parser)
    args = parser.parse_args()

    main(args)

    # metrics(args)
