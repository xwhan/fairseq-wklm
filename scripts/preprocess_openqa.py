from drqa.retriever import utils
from drqa import retriever, tokenizers
import json
import os
from drqa.tokenizers import SimpleTokenizer
import numpy as np
from tqdm import tqdm
import sys
from multiprocessing import Pool

sys.path.append('/private/home/xwhan/fairseq-py')
from fairseq.tokenization import BertTokenizer
"""
process SearchQA and TriviaQA data for BERT Ranker and BERT Reader
"""

dataset_path = "/private/home/xwhan/DrQA/data/datasets/data/datasets"


PROCESS_TOK = SimpleTokenizer()

def para_has_answer(para, answer, match='string'):
    global PROCESS_TOK
    para = utils.normalize(para)
    if match == 'string':
        para_text = PROCESS_TOK.tokenize(para).words(uncased=True)
        for single_answer in answer:
            single_answer = utils.normalize(single_answer)
            single_answer = PROCESS_TOK.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(para_text) - len(single_answer) + 1):
                if single_answer == para_text[i: i + len(single_answer)]:
                    return True
    elif match == 'regex':
        single_answer = utils.normalize(answer[0])
        if regex_match(para, single_answer):
            return True
    return False

def hash_q_id(question):
    return hashlib.md5(question.encode()).hexdigest()

def combine(doc, limit=384):
    curr = []
    curr_len = 0
    for split in regex.split(r'\n+', doc):
        split = split.strip()
        if len(split) == 0:
            continue
        # Maybe group paragraphs together until we hit a length limit
        if len(curr) > 0 and curr_len + len(split.split()) > limit:
            yield ' '.join(curr)
            curr = []
            curr_len = 0
        curr.append(split)
        curr_len += len(split.split())
    if len(curr) > 0:
        yield ' '.join(curr)



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

    orig_to_tok_index = []  # original token to wordpiece index
    tok_to_orig_index = []  # wordpiece token to original token index
    all_doc_tokens = []  # wordpiece tokens
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)  # wordpiece tokens
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    return orig_to_tok_index, tok_to_orig_index, doc_tokens, all_doc_tokens

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def read_data(data_name):
    for split in ['train','dev']:
        data = open(os.path.join(dataset_path, data_name, f'{split}.json')).readlines()
        answer_data = open(os.path.join(
            dataset_path, data_name, f'{split}.txt')).readlines()

        qa_train = [] # distant-supervision QA training
        recall = []
        for idx, line in enumerate(tqdm(data)):
            # each line is a QA pair, includes multiple paragraphs
            item = json.loads(line.strip())
            answer = json.loads(answer_data[idx].strip())
            
            gold = answer['answers']
            question = answer['question']
            covered = []
            for para in item:
                if para_has_answer(" ".join(para["document"]), gold):
                    covered.append(1)
                    qid, para_id = para['id']
                    qa_train.append(
                        {"q": question, "para": " ".join(para["document"]), "answer": gold, "qid": qid, "para_id":para_id })
                else:
                    covered.append(0)
            recall.append(np.sum(covered) > 0)
        
        print(f'{split} recall is {np.mean(recall)}')
        with open(os.path.join(dataset_path, data_name, "splits", f'{split}.json'), 'w') as f:
            for _ in qa_train:
                f.write(json.dumps(_) + "\n")


def read_data_for_ranking(data_name):
    """
    binary classification paragraph ranking
    """

    max_neg_samples = 20
    for split in ['train', 'dev']:
        data = open(os.path.join(dataset_path, data_name,
                                 f'{split}.json')).readlines()
        answer_data = open(os.path.join(
            dataset_path, data_name, f'{split}.txt')).readlines()

        qa_train = []  # distant-supervision QA training
        recall = []
        for idx, line in enumerate(tqdm(data)):
            # each line is a QA pair, includes multiple paragraphs
            item = json.loads(line.strip())
            answer = json.loads(answer_data[idx].strip())

            gold = answer['answers']
            question = answer['question']
            covered = []
            neg_num = 0
            for para in item:
                if para_has_answer(" ".join(para["document"]), gold):
                    covered.append(1)
                    qid, para_id = para['id']
                    qa_train.append(
                        {"q": question, "para": " ".join(para["document"]), "answer": gold, "qid": qid, "para_id": para_id, 'label':1})
                else:
                    if neg_num < max_neg_samples:
                        covered.append(0)
                        qid, para_id = para['id']
                        qa_train.append(
                            {"q": question, "para": " ".join(para["document"]), "answer": gold, "qid": qid, "para_id": para_id, 'label': 0})
                        neg_num += 1

            recall.append(np.sum(covered) > 0)

        print(f'{split} recall is {np.mean(recall)}')
        with open(os.path.join(dataset_path, f'{data_name}_ranking', "splits", f'{split}.json'), 'w') as f:
            for _ in qa_train:
                f.write(json.dumps(_) + "\n")


def process_raw_items(items, tokenizer):

    samples = []
    for d, qa in tqdm(items):
        
        d = json.loads(d.strip())  # list of documents
        qa = json.loads(qa.strip())
        gold = qa['answers']
        question = qa['question']
        question_toks = tokenizer.tokenize(question)

        for d_item in d:
            sample = {}
            qid, para_id = d_item['id']
            para = " ".join(d_item["document"])
            orig_to_tok_index, tok_to_orig_index, doc_tokens, wp_tokens = build_map(para, tokenizer)
            sample['tok_to_orig_index'] = tok_to_orig_index
            sample['para_subtoks'] = wp_tokens
            sample['para_toks'] = doc_tokens
            sample['q_subtoks'] = question_toks
            sample['qid'] = qid
            sample['score'] = 0.0
            sample['para_id'] = para_id
            sample['q'] = question
            sample['para'] = para
            sample['answer'] = gold
            sample['para_has_answer'] = int(para_has_answer(para, gold))
            samples.append(sample)
    return samples

def process_eval(data_name, split='valid'):
    """
    process data for final QA evaluation
    """
    tokenizer = BertTokenizer(
        '/private/home/xwhan/fairseq-py/vocab_dicts/vocab.txt', do_lower_case=True)
    data = open(os.path.join(dataset_path, data_name,
                             f'{split}.json')).readlines()
    answer_data = open(os.path.join(
            dataset_path, data_name, f'{split}.txt')).readlines()
    assert len(data) == len(answer_data)
    
    raw_data = list(zip(data, answer_data))
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

    with open(os.path.join(dataset_path, data_name, f'{split}_eval.json'), "w") as g:
        for s in samples:
            g.write(json.dumps(s) + '\n')


if __name__ == "__main__":
    data_name = sys.argv[1]
    # read_data(data_name)
    # process_eval(data_name)

    read_data_for_ranking(data_name)
