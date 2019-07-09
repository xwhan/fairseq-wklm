#!/usr/bin/env python3
# preprocessing code for span style question answering


import argparse
import json
import os
import string
import re
import sys
import numpy as np
from tqdm import tqdm
import nltk
import random
from joblib import Parallel, delayed

sys.path.append('/private/home/xwhan/fairseq-py')

# from fairseq.data.masked_lm_dictionary import BertDictionary
from fairseq import utils
from fairseq.tokenization import BertTokenizer, whitespace_tokenize


def add_span_and_char_offset(data_folder='/private/home/xwhan/dataset/webqsp/full'):
    documents = open(os.path.join(data_folder, 'documents.json')).readlines()
    documents = [json.loads(d) for d in documents]
    for d in tqdm(documents):
        doc_tokens = nltk.word_tokenize(d['document']['text'])
        for ent in d['document']['entities']:
            # ent_start_token = doc_tokens[ent['start']].strip()
            ent_tokens = doc_tokens[ent['start']:ent['end']]
            ent_text = " ".join(ent_tokens).strip()
            if ent_text == "":
                continue
            char_offset = d['document']['text'].find(ent_text)
            if char_offset == -1:
                ent_start_token = doc_tokens[ent['start']].strip()
                char_offset = d['document']['text'].find(ent_start_token)
            ent['char_start'] = char_offset
            ent['span_text'] = ent_text
    with open(os.path.join(data_folder, 'documents.json'), 'w') as f:
        for d in documents:
            f.write(json.dumps(d) + '\n')

def webqsp(data_folder='/private/home/xwhan/dataset/webqsp/full'):
    """
    question:
    answer
    """
    splits = ['train', 'dev', 'test']
    documents = open(os.path.join(data_folder, 'documents.json')).readlines()
    documents = [json.loads(d) for d in documents]
    id2doc = {d['documentId']:d for d in documents}

    tokenizer = BertTokenizer('/private/home/xwhan/fairseq-py/vocab_dicts/vocab.txt')

    def check_ans_coverage(data, id2doc):
        covered = []
        for item in tqdm(data):
            ans_m_ids = [ans['kb_id'] for ans in item['answers']]
            passage_ids = [p['document_id'] for p in item['passages']]
            candidate = set()
            for p_id in passage_ids:
                if p_id not in id2doc:
                    continue
                for ent in id2doc[p_id]['document']['entities']:
                    candidate.add(ent['text'])
            ans_in_doc = False
            for mid in ans_m_ids:
                ans_in_doc = ans_in_doc or (mid in candidate)
            covered.append(int(ans_in_doc))
        print(np.mean(covered))

    def process(s, tokenizer):
        try:
            return tokenizer.tokenize(s.lower())
        except:
            print('failed on', s)
            raise

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def find_ent_span(ent, current_sent, char_to_word_offset, doc_tokens, all_doc_tokens, char_pos, orig_to_tok_index):

        if ent['char_start'] == -1:
            return (-1, -1)

        if char_pos == 0:
            ent_offset = char_pos + ent['char_start']
        else: 
            ent_offset = char_pos + 1 + ent['char_start']
        orig_ent_text = ent['span_text']
        ent_length = len(orig_ent_text)
        start_position = char_to_word_offset[ent_offset]
        end_position = char_to_word_offset[min(ent_offset + ent_length - 1, len(char_to_word_offset) - 1)]
        tok_start_position = orig_to_tok_index[start_position]
        if end_position < len(doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        # (tok_start_position, tok_end_position) = _improve_answer_span(all_doc_tokens, tok_start_position, tok_end_position, process, orig_ent_text)

        return tok_start_position, tok_end_position

    def _process_sample(item, id2doc, tokenizer):

        ans_m_ids = [ans['kb_id'] for ans in item['answers']]
        passage_ids = [p['document_id'] for p in item['passages']]
        passages = [id2doc[pid] for pid in passage_ids if pid in id2doc]
        passages = [p for p in passages if len(p['document']['entities'])!=0]
        passages = passages[:25]
        context = " ".join([passage['document']['text'] for passage in passages])
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
        orig_to_tok_index = []
        tok_to_orig_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = process(token, tokenizer) # wordpiece tokens
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        q = process(item['question'], tokenizer)
        answer_start = []
        answer_end = []
        candidate_start = []
        candidate_end = []
        is_impossible = True
        char_pos = 0
        answer_label = []
        for passage in passages:
            text = passage['document']['text']
            for ent in passage['document']['entities']:
                ent_mid = ent['text']
                start_wp, end_wp = find_ent_span(ent, text, char_to_word_offset, doc_tokens, all_doc_tokens, char_pos, orig_to_tok_index)
                if start_wp != -1:
                    candidate_start.append(start_wp)
                    candidate_end.append(end_wp)
                if ent_mid in ans_m_ids:
                    if start_wp != -1:
                        answer_start.append(start_wp)
                        answer_end.append(end_wp)
                        is_impossible = False
                        answer_label.append(ent['name'])

            if char_pos > 0:
                char_pos += 1 + len(text) # space between docs
            else:
                char_pos += len(text)

        if is_impossible:
            answer_start.append(-1)
            answer_end.append(-1)


        return (all_doc_tokens, q, answer_start, answer_end, answer_label, candidate_start, candidate_end, is_impossible)

    for split in splits:

        if not os.path.exists(os.path.join(data_folder, split)):
            os.makedirs(os.path.join(data_folder, split))

        question_out = open(os.path.join(data_folder, split, 'q.txt'), 'w')
        context_out = open(os.path.join(data_folder, split, 'c.txt'), 'w')
        answer_start_out = open(os.path.join(data_folder, split, 'ans_start.txt'), 'w')
        answer_end_out = open(os.path.join(data_folder, split, 'ans_end.txt'), 'w')
        # answer_out = open(os.path.join(data_folder, split, 'ans.txt'), 'w')


        data = open(os.path.join(data_folder, f'{split}.json')).readlines()
        data = [json.loads(_) for _ in data]
        # check_ans_coverage(data, id2doc)

        samples = Parallel(n_jobs=40)(delayed(_process_sample)(item, id2doc, tokenizer) for item in tqdm(data))

        for sample in samples:
            all_doc_tokens, q, answer_start, answer_end, answer_label, candidate_start, candidate_end, is_impossible = sample

            print(' '.join(all_doc_tokens), file=context_out)
            print(' '.join(q), file=question_out)
            print(' '.join([str(ii) for ii in answer_start]), file=answer_start_out)
            print(' '.join([str(ii) for ii in answer_end]), file=answer_end_out)

        # print(f'Unanswerable questions {num_unanswerable}')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--inputs',
        required=True,
        nargs='+',
        help='files to process.',
    )
    parser.add_argument(
        '--output',
        required=True,
        metavar='DIR',
        help='Path for output',
    )

    parser.add_argument('--dict_path', default='/private/home/xwhan/fairseq-py/vocab_dicts/dict.txt')
    parser.add_argument('--vocab_path', default='/private/home/xwhan/fairseq-py/vocab_dicts/vocab.txt')

    args = parser.parse_args()
    utils.print_args(args)

    webqsp()


def _improve_answer_span(doc_tokens, input_start, input_end, process,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(process(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


if __name__ == '__main__':
    # add_span_and_char_offset()

    main()
