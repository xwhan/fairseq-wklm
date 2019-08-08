import argparse
import json
import os
import sys
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.append('/private/home/xwhan/fairseq-py')

from fairseq.data.masked_lm_dictionary import BertDictionary
from fairseq import utils
from fairseq.tokenization import BertTokenizer, whitespace_tokenize


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def process(s, tokenizer):
    try:
        return tokenizer.tokenize(s)
    except:
        print('failed on', s)
        raise

def process_file(folder, output, use_ent_marker=False):
    """
    sentence
    e1_start
    e1_len
    e2_start
    e2_len
    label
    """

    tokenizer = BertTokenizer('/private/home/xwhan/fairseq-py/vocab_dicts/vocab.txt')
    relation2id = json.load(open(os.path.join(folder, "relation2id.json")))

    if use_ent_marker:
        e1_start_marker = "[unused0]"
        e1_end_marker = "[unused1]"
        e2_start_marker = "[unused2]"
        e2_end_marker = "[unused3]"

    for split in  ['train', 'valid', 'test']:
        file_path = os.path.join(folder, f'{split}.json')
        with open(file_path, "r", encoding='utf-8') as reader:
            data = json.load(reader)
        sent_out = open(os.path.join(output, split, 'sent.txt'), 'w')
        lbl_out = open(os.path.join(output, split, 'lbl.txt'), 'w')
        e1_start_out = open(os.path.join(output, split, 'e1_start.txt'), 'w')
        e2_start_out = open(os.path.join(output, split, 'e2_start.txt'), 'w')
        e1_len_out = open(os.path.join(output, split, 'e1_len.txt'), 'w')
        e2_len_out = open(os.path.join(output, split, 'e2_len.txt'), 'w')

        for item in data:
            sent_toks = item['token']
            lbl_rel = item['relation']
            wp_toks = []
            orig_to_tok_index = []
            for i, tok in enumerate(sent_toks):
                orig_to_tok_index.append(len(wp_toks))
                sub_toks = process(tok.lower(), tokenizer)
                for sub_tok in sub_toks:
                    wp_toks.append(sub_tok)
            
            e1_start = orig_to_tok_index[item['subj_start']] 
            e1_end = orig_to_tok_index[item['subj_end'] + 1] if item['subj_end'] + 1 < len(orig_to_tok_index) else len(orig_to_tok_index)
            e2_start = orig_to_tok_index[item['obj_start']]
            e2_end = orig_to_tok_index[item['obj_end'] + 1] if item['obj_end'] + 1 < len(orig_to_tok_index) else len(orig_to_tok_index)

            e1_len = e1_end - e1_start
            e2_len = e2_end - e2_start

            if self.use_ent_marker:
                e1_len += 2
                e2_len += 2
                if e1_start < e2_start:
                    assert e1_end < e2_start
                    wp_toks = wp_toks[:e1_start] + [e1_start_marker] + wp_toks[e1_start:e1_end+1] + [e1_end_marker] + wp_toks[e1_end+1:e2_start] + [e2_start_marker] + wp_toks[e2_start:e2_end+1] + [e2_end_marker] + wp_toks[e2_end+1:]
                    e1_end += 1
                    e2_start += 2
                    e2_end += 3
                else:
                    assert e2_end < e1_start
                    wp_toks = wp_toks[:e2_start] + [e2_start_marker] + wp_toks[e2_start:e2_end+1] + [e2_end_marker] + wp_toks[e2_end+1:e1_start] + [e1_start_marker] + wp_toks[e1_start:e1_end+1] + [e1_end_marker] + wp_toks[e1_end+1:]
                    e2_end += 1
                    e1_start += 2
                    e1_end += 3

            lbl = relation2id[lbl_rel]
            print(" ".join(wp_toks), file=sent_out)
            print(lbl, file=lbl_out)
            print(e1_start, file=e1_start_out)
            print(e2_start, file=e2_start_out)
            print(e1_len, file=e1_len_out)
            print(e2_len, file=e2_len_out)


if __name__ == '__main__':
    process_file("/private/home/xwhan/dataset/tacred/raw", "/private/home/xwhan/dataset/tacred/processed-splits", use_ent_marker=True)