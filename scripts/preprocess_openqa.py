from drqa.retriever import utils
from drqa import retriever, tokenizers
import json
import os
from drqa.tokenizers import SimpleTokenizer
import numpy as np
from tqdm import tqdm
import sys

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

def process_eval(data_name, split='valid'):
    """
    process data for final QA evaluation
    """
    pass

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

if __name__ == "__main__":
    data_name = sys.argv[1]
    read_data(data_name)
