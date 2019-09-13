
import argparse
import json
import os
import sys
from tqdm import tqdm

sys.path.append('/private/home/xwhan/fairseq-py')
from fairseq.tokenization import BertTokenizer

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

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def char_to_word(context):
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
    return doc_tokens, char_to_word_offset

def process(s, tokenizer):
    try:
        return tokenizer.tokenize(s)
    except:
        print('failed on', s)
        raise

def process_file(folder, output):
    tokenizer = BertTokenizer(
        '/private/home/xwhan/fairseq-py/vocab_dicts/vocab.txt', do_lower_case=True)

    if os.path.exists(os.path.join(output, "label2id.json")):
        label2ids = json.load(open(os.path.join(output, "label2id.json")))
    else:
        import pdb;pdb.set_trace()
        label2ids = {}
    
    for split in ["train", "valid", "test"]:
        file_path = os.path.join(folder,  f'{split}.json')
        with open(file_path, "r", encoding='utf-8') as reader:
            data = json.load(reader)
        
        sent_out = open(os.path.join(output, split, 'sent.txt'), 'w')
        lbl_out = open(os.path.join(output, split, 'lbl.txt'), 'w')
        e_start_out = open(os.path.join(output, split, 'e_start.txt'), 'w')
        e_len_out = open(os.path.join(output, split, 'e_len.txt'), 'w')

        for item in tqdm(data):
            sent = item["sent"]
            e_char_start = item["start"]
            e_char_end = item["end"] - 1
            sent_toks, char_to_word_offset = char_to_word(sent)
            labels = item["labels"]

            if len(labels) == 0:
                print(item)

            lbls = []
            for l in labels:
                if l not in label2ids:
                    label2ids[l] = len(label2ids)
                    lbls.append(label2ids[l])
                else:
                    lbls.append(label2ids[l])

            orig_to_tok_index = []
            sent_wp_toks = []
            for i, token in enumerate(sent_toks):
                orig_to_tok_index.append(len(sent_wp_toks))
                token = convert_token(token)
                sub_tokens = process(token, tokenizer)
                for sub_token in sub_tokens:
                    sent_wp_toks.append(sub_token)

            e_start_word = char_to_word_offset[e_char_start]
            e_start = orig_to_tok_index[e_start_word]
            e_end_word = char_to_word_offset[e_char_end] + 1
            e_end = orig_to_tok_index[e_end_word] if e_end_word < len(
                orig_to_tok_index) else len(sent_wp_toks)

            # if e_end_word >= len(orig_to_tok_index):
            #     print(sent_wp_toks[e_start:e_end])
            #     print(sent[e_char_start:e_char_end+1])
            #     import pdb
            #     pdb.set_trace()

            e_len = e_end - e_start
            
            print(" ".join(sent_wp_toks), file=sent_out)
            print(" ".join([str(l) for l in lbls]), file=lbl_out)
            print(e_start, file=e_start_out)
            print(e_len, file=e_len_out)
    
    json.dump(label2ids, open(os.path.join(output, 'label2id.json'), 'w'))


def process_onto(folder, output):
    tokenizer = BertTokenizer(
        '/private/home/xwhan/fairseq-py/vocab_dicts/vocab.txt', do_lower_case=True)

    if os.path.exists(os.path.join(output, "label2id.json")):
        label2ids = json.load(open(os.path.join(output, "label2id.json")))
    else:
        label2ids = {}

    for split in ["train", "valid", "test"]:
        file_path = os.path.join(folder,  f'{split}.json')
        with open(file_path, "r", encoding='utf-8') as reader:
            data = [json.loads(line.strip()) for line in reader.readlines()]
        
        sent_out = open(os.path.join(output, split, 'sent.txt'), 'w')
        lbl_out = open(os.path.join(output, split, 'lbl.txt'), 'w')
        e_start_out = open(os.path.join(output, split, 'e_start.txt'), 'w')
        e_len_out = open(os.path.join(output, split, 'e_len.txt'), 'w')

        for item in tqdm(data):
            left = " ".join(item["left_context_token"]).strip()
            right = " ".join(item["right_context_token"]).strip()
            mention = item["mention_span"]

            if len(left) == 0:
                e_char_start = 0
                e_char_end = len(mention) - 1
                sent = mention
            else:
                e_char_start = len(left) + 1
                e_char_end = len(left) + 1 + len(mention) - 1
                sent = left + " " + mention

            if len(right) != 0:
                sent = sent + " " + right

            sent_toks, char_to_word_offset = char_to_word(sent)
            labels = item["y_str"]

            lbls = []
            for l in labels:
                if l not in label2ids:
                    label2ids[l] = len(label2ids)
                    lbls.append(label2ids[l])
                else:
                    lbls.append(label2ids[l])

            orig_to_tok_index = []
            sent_wp_toks = []
            for i, token in enumerate(sent_toks):
                orig_to_tok_index.append(len(sent_wp_toks))
                token = convert_token(token)
                sub_tokens = process(token, tokenizer)
                for sub_token in sub_tokens:
                    sent_wp_toks.append(sub_token)

            e_start_word = char_to_word_offset[e_char_start]
            e_start = orig_to_tok_index[e_start_word]
            e_end_word = char_to_word_offset[e_char_end] + 1
            e_end = orig_to_tok_index[e_end_word] if e_end_word < len(
                orig_to_tok_index) else len(sent_wp_toks)



            e_len = e_end - e_start

            print(" ".join(sent_wp_toks), file=sent_out)
            print(" ".join([str(l) for l in lbls]), file=lbl_out)
            print(e_start, file=e_start_out)
            print(e_len, file=e_len_out)
        
    json.dump(label2ids, open(os.path.join(output, 'label2id.json'), 'w'))

if __name__ == '__main__':
    process_file("/private/home/xwhan/dataset/FIGER", "/private/home/xwhan/dataset/FIGER/processed-splits")
    # process_onto("/private/home/xwhan/dataset/ontonotes",
    #              "/private/home/xwhan/dataset/ontonotes/processed-splits")


