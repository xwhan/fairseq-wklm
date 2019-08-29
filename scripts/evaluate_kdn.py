import json
import sys
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from collections import defaultdict
import os

sys.path.append('/private/home/xwhan/fairseq-py')

from fairseq import checkpoint_utils, tasks, options
from fairseq.tokenization import BertTokenizer
import numpy as np

data_path = "/private/home/xwhan/KBs/Wikidata/fact_completion/text_statement_tokenized.json"

def debinarize_list(vocab, indices):
    return [vocab[idx] for idx in indices]

def bert_batcher(bert_tokenizer, pretext, ground, candidate_dict, task, bsz=1000):
    """
    candidate_dict: {'text': 'binarized_idx'}
    """
    indexed_pretext = binarize_list(task.dictionary, pretext)
    indexed_pretext = [task.dictionary.cls()] + indexed_pretext
    pretext_len = len(indexed_pretext)

    candidate_lens = len(candidate_dict)
    candidate_texts = list(candidate_dict.keys())
    max_len = max([len(v) for k, v in candidate_dict.items()])

    for batch_id in range(0, candidate_lens, bsz):
        candidate_batch = candidate_texts[batch_id: batch_id + bsz]
        labels = [int(c.strip().lower() == ground.strip().lower()) for c in candidate_batch]
        real_bsz = len(candidate_batch)
        lens = []
        inputs = np.full((real_bsz, max_len + pretext_len), 0)
        segments = np.full((real_bsz, max_len + pretext_len), 0)
        entity_masks = np.full((real_bsz, 1, max_len + pretext_len), 0, dtype=float)

        indexed_candidate = []
        for idx, c_txt in enumerate(candidate_batch):
            indexed_c = candidate_dict[c_txt]
            indexed_candidate.append(indexed_c)
            lens.append(pretext_len + len(indexed_c))
            inputs[idx,:(pretext_len + len(indexed_c))] = indexed_pretext + indexed_c
            
            # use start and end toks
            entity_masks[idx, 0, pretext_len] = 1
            entity_masks[idx, 0, pretext_len + len(indexed_c) - 2] = 2
            
            # boundary masks
            entity_masks[idx, 0, pretext_len-1] = -1
            entity_masks[idx, 0, pretext_len + len(indexed_c) - 1] = -2

        yield {'labels': labels, 'inputs': inputs, 'lens': lens, 'pretext_len': pretext_len, 'indexed_candidates': indexed_candidate, 'segments': segments, 'entity_masks': entity_masks}


def _load_models(args, task):
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.model_path.split(':'), arg_overrides=vars(args)
        ,task=task
    )
    for model in models:
        model.eval()
    return models[0]

def hits(label_and_scores, ks = [5,10], alpha=0):
    ranked = sorted(label_and_scores, key=lambda x:x[1] + alpha*x[2], reverse=True)
    hits = {}
    for k in ks:
        topk = ranked[:k]
        topk_labels = [_[0] for _ in topk]
        hits[str(k)] = float(np.sum(topk_labels) > 0)
    return hits

def binarize_list(vocab, words):
    return [vocab.index(w) for w in words]

def bert_eval(model, task, k=0):
    bert_tokenizer = BertTokenizer(os.path.join(
        args.data, 'vocab.txt'), do_lower_case=True)
    statements = json.load(open(data_path))
    metrics = defaultdict(dict)
    # for rel in statements.keys():
    
    rel = list(statements.keys())[k]

    tail_candidates = statements[rel]['candidates']
    tokenized_candidates = statements[rel]['tokenized_candidates']

    tokenized_pretexts = [s['pretext_toks'] for s in statements[rel]['statements']]
    true_tails = [s['answer'] for s in statements[rel]['statements']]

    # tokenize all candidate
    candidate_dict = {}
    for idx in range(len(tokenized_candidates)):
        candidate_dict[tail_candidates[idx]] = binarize_list(task.dictionary, tokenized_candidates[idx]) + [task.dictionary.sep()]

    rel_hits = defaultdict(list)
    with torch.no_grad():
        for text, ground in tqdm(list(zip(tokenized_pretexts, true_tails))):
            label_and_scores = []
            for batch in bert_batcher(bert_tokenizer, text, ground, candidate_dict, task):
                cuda_input = torch.from_numpy(batch['inputs']).to('cuda')
                cuda_segment = torch.from_numpy(batch['segments']).to('cuda')
                cuda_entity_masks = torch.from_numpy(batch['entity_masks']).to('cuda')
                entity_logits, lm_logits = model(cuda_input, cuda_segment, cuda_entity_masks)
                indexed_candidates = batch['indexed_candidates']
            
                batch_probs = F.softmax(entity_logits, dim=-1).squeeze(1).cpu().numpy()
                batch_lm_probs = F.softmax(lm_logits, dim=-1).cpu().numpy()

                pretext_len = batch['pretext_len']
                lens = batch['lens']
                token_lm_scores = []
                for idx, length in enumerate(lens):
                    lmprobs = batch_lm_probs[idx, :, :]
                    cand_probs = lmprobs[pretext_len:length,:]
                    c_len = length - pretext_len
                    token_probs = cand_probs[np.arange(c_len), indexed_candidates[idx]]
                    first_token_prob = token_probs[0]
                    token_lm_scores.append(first_token_prob)

                scores = list(batch_probs[:,1])
                labels = batch["labels"]
                label_and_scores += list(zip(labels, scores, token_lm_scores))
            item_hits = hits(label_and_scores)
            for k, v in item_hits.items():
                rel_hits[k].append(v)
    for k in rel_hits.keys():
        metrics[rel][k] = np.mean(rel_hits[k])
    print(f'{rel}', metrics[rel])

    # print(f'\nresults:', metrics)
    return metrics

def pre_tokenize(path):
    """
    tokenized the evaluation data for faster evaluation
    """
    tokenizer = BertTokenizer(
        '/private/home/xwhan/fairseq-py/vocab_dicts/vocab.txt', do_lower_case=True)
    fact_data = json.load(open(path))
    for k, content in fact_data.items():
        tokenized_candidates = [tokenizer.tokenize(c) for c in content['candidates']]
        content['tokenized_candidates'] = tokenized_candidates
        for s in content['statements']:
            pretext = s['statement'].split('<o>')[0].strip()
            answer_text = s['statement'].split('<o>')[1].strip()
            s['pretext_toks'] = tokenizer.tokenize(pretext)
            s['answer'] = answer_text
    json.dump(fact_data, open('/private/home/xwhan/KBs/Wikidata/fact_completion/text_statement_tokenized.json', 'w'))

def main(args):
    task = tasks.setup_task(args)
    model = _load_models(args, task)
    model.half()
    model.eval()
    model.cuda()
    bert_eval(model, task, args.rel_id)

if __name__ == '__main__':
    parser = options.get_parser('Trainer', 'kdn')
    options.add_dataset_args(parser)
    parser.add_argument('--criterion', default='kdn_loss')
    parser.add_argument('--model-path', metavar='FILE', help='path(s) to model file(s), colon separated', default='/checkpoint/xwhan/2019-08-16/kdn_v2_boundary.adam.bert.crs_ent.seed3.bsz4.0.01.lr1e-05.ngpu32/checkpoint_best.pt')
    parser.add_argument('--rel-id', default=0, type=int, help="which relation to evaluate")
    args = options.parse_args_and_arch(parser)
    args = parser.parse_args()
    main(args)
