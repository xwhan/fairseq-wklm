
import numpy as np
import json
import sys
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from collections import defaultdict
import os

sys.path.append('/private/home/xwhan/fairseq-py')
from fairseq.tokenization import BertTokenizer
from fairseq import checkpoint_utils, tasks, options

from pytorch_transformers import BertTokenizer as HuggBertTokenizer
cased_tokenizer = HuggBertTokenizer.from_pretrained("bert-base-cased")

data_path = "/private/home/xwhan/LAMA-ACL2019/datasets/Google_RE/google_re_tokenized.json"

# data_path = "/private/home/xwhan/LAMA-ACL2019/datasets/TREX/google_re_tokenized.json"
trex_meta = "/private/home/fabiopetroni/LAMA/data/relations_trex.jsonl"
common_vocab = "/private/home/xwhan/LAMA/data/common_vocab_cased.txt"

def debinarize_list(vocab, indices):
    return [vocab[idx] for idx in indices]

def load_vocab_set():
    words = [w.strip() for w in open(common_vocab).readlines()]
    return set(words)

def bert_batcher(bert_tokenizer, left_text, right_text, ground, candidate_dict, task, bsz=1000):
    """
    candidate_dict: {'text': 'binarized_idx'}
    """
    indexed_left_text = binarize_list(task.dictionary, left_text)
    indexed_left_text = [task.dictionary.cls()] + indexed_left_text

    indexed_right_text = binarize_list(task.dictionary, right_text)
    indexed_right_text = indexed_right_text + [task.dictionary.sep()]

    left_len = len(indexed_left_text)
    right_len = len(indexed_right_text)

    candidate_nums = len(candidate_dict)
    candidate_texts = list(candidate_dict.keys())
    max_len = max([len(v) for k, v in candidate_dict.items()]) # max length of candidates

    for batch_id in range(0, candidate_nums, bsz):
        candidate_batch = candidate_texts[batch_id: batch_id + bsz]
        labels = [int(c.strip().lower() == ground.strip().lower())
                  for c in candidate_batch]
        real_bsz = len(candidate_batch)
        lens = []
        inputs = np.full((real_bsz, max_len + left_len + right_len), 0)
        segments = np.full((real_bsz, max_len + left_len + right_len), 0)
        entity_masks = np.full((real_bsz, 1, max_len + left_len + right_len), 0, dtype=float)

        indexed_candidate = []
        for idx, c_txt in enumerate(candidate_batch):
            indexed_c = candidate_dict[c_txt]
            indexed_candidate.append(indexed_c)
            lens.append(left_len + len(indexed_c) + right_len)
            # inputs[idx, :(left_len + len(indexed_c) + right_len)] = indexed_left_text + indexed_c + indexed_right_text

            inputs[idx, :(left_len + len(indexed_c) + right_len)
                   ] = indexed_left_text + [task.dictionary.index("[MASK]")] + indexed_right_text

            assert len(indexed_c) == 1

            # boundary masks
            entity_masks[idx, 0, left_len-1] = -1
            entity_masks[idx, 0, left_len + len(indexed_c)] = -2

        yield {'labels': labels, 'inputs': inputs, 'lens': lens, 'pretext_len': left_len, 'indexed_candidates': indexed_candidate, 'segments': segments, 'entity_masks': entity_masks}

def _load_models(args, task):
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.model_path.split(':'), arg_overrides=vars(args), task=task
    )
    for model in models:
        model.eval()
    return models[0]

def hits(label_and_scores, ks=[1, 5, 10], alpha=0):
    # ranked = sorted(label_and_scores, key=lambda x: x[1] + alpha*x[2], reverse=True)
    ranked = sorted(label_and_scores, key=lambda x: x[2], reverse=True)
    hits = {}
    for k in ks:
        topk = ranked[:k]
        topk_labels = [_[0] for _ in topk]
        hits[str(k)] = float(np.sum(topk_labels) > 0)
    return hits

def binarize_list(vocab, words):
    return [vocab.index(w) for w in words]

def parse_template(template, subject_label, object_label):
    SUBJ_SYMBOL = "[X]"
    OBJ_SYMBOL = "[Y]"
    template = template.replace(SUBJ_SYMBOL, subject_label)
    template = template.replace(OBJ_SYMBOL, object_label)
    return template

def process_googlere(path="/private/home/xwhan/LAMA-ACL2019/datasets/Google_RE"):

    vocab_set = load_vocab_set()

    if path.endswith("Google_RE"):
        templates = {
            "/people/person/place_of_birth": "[X] was born in [Y] .",
            "/people/deceased_person/place_of_death": "[X] died in [Y] .",
            "/people/person/date_of_birth": "[X] (born [Y])."
        }
    else:
        trex_templates = [json.loads(ii.strip()) for ii in open(trex_meta).readlines()]
        templates = {ii["relation"]: ii["template"] for ii in trex_templates}

    bert_tokenizer = BertTokenizer(os.path.join(args.data, 'vocab.txt'), do_lower_case=True)
    outputs = {}
    for file in os.listdir(path):
        if not file.endswith("jsonl"):
            continue
        data = [json.loads(ii.strip()) for ii in open(os.path.join(path, file)).readlines()]
        statements = []
        candidates = {}
        if "pred" in data[0]:
            relation = data[0]["pred"]
        else:
            relation = data[0]["predicate_id"]
        candidates_cased = {}
        for item in data:
            assert len(item['obj_label'].split(" ")) == 1
            item['obj_label'] = item['obj_label'].strip()

            # filter some of the examples
            if item['obj_label'] not in vocab_set:
                continue

            if "judgements" in item:
                num_no = 0
                num_yes = 0
                for x in item['judgments']:
                    if (x['judgment']=="yes"):
                        num_yes+=1
                    else:
                        num_no+=1
                if num_no > num_yes:
                    continue

            if item['obj_label'] not in candidates:
                candidates[item['obj_label']] = bert_tokenizer.tokenize(
                    item['obj_label'])
                candidates_cased[item['obj_label']] = cased_tokenizer.tokenize(
                    item['obj_label'])
            sub_label = item['sub_label'].strip()
            masked_sentence = parse_template(templates[relation], sub_label, "[MASK]")
            mask_offset = masked_sentence.find("[MASK]")
            left_context = masked_sentence[:mask_offset]
            right_context = masked_sentence[mask_offset + len("[MASK]"):]

            gold = item["obj_label"]
            tokenized_left = bert_tokenizer.tokenize(left_context)
            tokenized_right = bert_tokenizer.tokenize(right_context)
            statements.append({"left_context": tokenized_left, "left_cased": cased_tokenizer.tokenize(
                left_context), "right_cased": cased_tokenizer.tokenize(right_context), "right_context": tokenized_right, "label": gold})
        
        print(f'{len(statements)} examples for relation {relation}')

        outputs[relation] = {'statements': statements, 'candidates': candidates, "candidates_cased": candidates_cased}
    json.dump(outputs, open(os.path.join(path, "google_re_tokenized.json"), "w"))

def bert_eval(model, task, k=0):
    bert_tokenizer = BertTokenizer(os.path.join(
        args.data, 'vocab.txt'), do_lower_case=True)
    statements = json.load(open(data_path))
    metrics = defaultdict(dict)

    rel = list(statements.keys())[k]
    print(f'evaluating for relation {rel} with {len(statements[rel]["statements"])} examples...')

    tail_candidates = statements[rel]['candidates'] # dict{answer: tokenized answer}
    tokenized_left = [s['left_context'] for s in statements[rel]['statements']]
    tokenized_right = [s['right_context'] for s in statements[rel]['statements']]
    true_tails = [s['label'] for s in statements[rel]['statements']] # groundtruth entity names

    # binarize all candidate eneitite
    candidate_dict = {}
    for k, v in tail_candidates.items():
        candidate_dict[k] = binarize_list(task.dictionary, tail_candidates[k])

    rel_hits = defaultdict(list)
    with torch.no_grad():
        for left_text, right_text, ground in tqdm(list(zip(tokenized_left, tokenized_right, true_tails))):
            label_and_scores = []
            for batch in bert_batcher(bert_tokenizer, left_text, right_text, ground, candidate_dict, task):
                cuda_input = torch.from_numpy(batch['inputs']).to('cuda')
                cuda_segment = torch.from_numpy(batch['segments']).to('cuda')
                cuda_entity_masks = torch.from_numpy(
                    batch['entity_masks']).to('cuda')
                entity_logits, lm_logits = model(cuda_input, cuda_segment, cuda_entity_masks)
                indexed_candidates = batch['indexed_candidates']

                batch_probs = F.softmax(entity_logits, dim=-1).squeeze(1).cpu().numpy()
                batch_lm_probs = F.log_softmax(lm_logits, dim=-1).cpu().numpy()

                pretext_len = batch['pretext_len']
                lens = batch['lens']
                token_lm_scores = []

                for idx, length in enumerate(lens):
                    logprobs = batch_lm_probs[idx, :, :]
                    first_token_prob = logprobs[pretext_len, indexed_candidates[idx][0]]
                    token_lm_scores.append(first_token_prob)

                scores = list(batch_probs[:, 1])
                labels = batch["labels"]
                label_and_scores += list(zip(labels, scores, token_lm_scores))
            item_hits = hits(label_and_scores)
            for k, v in item_hits.items():
                rel_hits[k].append(v)
    for k in rel_hits.keys():
        metrics[rel][k] = np.mean(rel_hits[k])
    print(f'{rel}', metrics[rel])

    return metrics

def main(args):
    task = tasks.setup_task(args)
    model = _load_models(args, task)
    # model.half()
    model.eval()
    model.cuda()
    model = nn.DataParallel(model)
    bert_eval(model, task, args.rel_id)

    # process_googlere()
    # process_googlere("/private/home/xwhan/LAMA-ACL2019/datasets/TREX")

if __name__ == '__main__':
    parser = options.get_parser('Trainer', 'kdn')
    options.add_dataset_args(parser)
    parser.add_argument('--criterion', default='kdn_loss')
    parser.add_argument('--model-path', metavar='FILE', help='path(s) to model file(s), colon separated', default='/checkpoint/xwhan/2019-08-29/kdn_v2_mask0.05.adam.bert.crs_ent.seed3.bsz4.0.01.lr1e-05.ngpu32/checkpoint_best.pt')
    parser.add_argument('--rel-id', default=0, type=int,
                        help="which relation to evaluate")
    args = options.parse_args_and_arch(parser)
    args = parser.parse_args()
    main(args)
