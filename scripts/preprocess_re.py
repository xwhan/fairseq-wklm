import argparse
import json
import os
import sys
from tqdm import tqdm

sys.path.append('/private/home/xwhan/fairseq-py')

from fairseq.tokenization import BertTokenizer


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

def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
            return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token

def process_file(folder, output, lower=True):
    """
    sentence
    e1_start
    e1_len
    e2_start
    e2_len
    label
    """
    if lower:
        tokenizer = BertTokenizer('/private/home/xwhan/fairseq-py/vocab_dicts/vocab.txt', do_lower_case=lower)
    else:
        tokenizer = BertTokenizer('/private/home/xwhan/fairseq-py/vocab_cased/vocab.txt')
    relation2id = json.load(open(os.path.join(folder, "relation2id.json")))
    special_tokens = {"[e1_start]": "[unused1]", "[e1_end]":"[unused2]", "[e2_start]": "[unused3]", "[e2_end]": "[unused4]"}

    def get_special_token(w):
        if w not in special_tokens:
            special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
        return special_tokens[w]


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
        ids_out = open(os.path.join(output, split, 'ids.txt'), 'w')
        e1_type_out = open(os.path.join(output, split, 'e1_type.txt'), 'w')
        e2_type_out = open(os.path.join(output, split, 'e2_type.txt'), 'w')

        for item in data:
            sent_toks = [convert_token(t) for t in item['token']]
            lbl_rel = item['relation']
            wp_toks = []
            orig_to_tok_index = []
            for i, tok in enumerate(sent_toks):
                orig_to_tok_index.append(len(wp_toks))
                sub_toks = process(tok, tokenizer)
                for sub_tok in sub_toks:
                    wp_toks.append(sub_tok)
            wp_toks.append("[SEP]")
            
            e1_start = orig_to_tok_index[item['subj_start']] 
            e1_end = orig_to_tok_index[item['subj_end'] + 1] if item['subj_end'] + 1 < len(orig_to_tok_index) else len(orig_to_tok_index) # the wp tok position after the entity
            e2_start = orig_to_tok_index[item['obj_start']]
            e2_end = orig_to_tok_index[item['obj_end'] + 1] if item['obj_end'] + 1 < len(orig_to_tok_index) else len(orig_to_tok_index)

            e1_len = e1_end - e1_start
            e2_len = e2_end - e2_start

            e1_type = get_special_token("SUBJ=%s" % item['subj_type'])
            e2_type = get_special_token("OBJ=%s" % item['obj_type'])


            lbl = relation2id[lbl_rel]
            print(" ".join(wp_toks), file=sent_out)
            print(lbl, file=lbl_out)
            print(e1_start, file=e1_start_out)
            print(e2_start, file=e2_start_out)
            print(e1_len, file=e1_len_out)
            print(e2_len, file=e2_len_out)
            print(item['id'], file=ids_out)
            print(e1_type, file=e1_type_out)
            print(e2_type, file=e2_type_out)

if __name__ == '__main__':
    # process_file("/private/home/xwhan/dataset/tacred/raw", "/private/home/xwhan/dataset/tacred/processed-splits")

    process_file("/private/home/xwhan/dataset/tacred/raw", "/private/home/xwhan/dataset/tacred/processed-splits-cased", lower=False)
