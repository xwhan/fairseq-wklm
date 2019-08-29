#!/usr/bin/env python3
# preprocessing code for span style question answering
"""
take json format files and create tokenized questions and contexts
"""


import argparse
import json
import os
import sys
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import Pool

sys.path.append('/private/home/xwhan/fairseq-py')

from fairseq.tokenization import BertTokenizer, whitespace_tokenize
from fairseq import utils
from fairseq.data.masked_lm_dictionary import BertDictionary

q_words = [["how", "many"], ["how", "long"], ["how"], ["which"], ["what"], ["when"], ["whose"], ["who"], ["where"], ["why"]]


def _process_samples(items, tokenizer):

    outputs = []
    for item in items:
        answer_list = [_.lower()
                        for _ in item['answer']]  # multiple answers
        context = item['para'].lower()
        sample_id = item['qid'] + "_para_id" + str(item['para_id'])
        # process context
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
        all_doc_tokens = []
        for (i, token) in enumerate(doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = process(token, tokenizer)  # wordpiece tokens
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
    
        q = process(item['q'].lower(), tokenizer)

        answer_start = []
        answer_end = []
        answer_text = []
        is_impossible = True
        for answer in answer_list:
            start, end = find_ans_span(
                answer, context, char_to_word_offset, doc_tokens, all_doc_tokens, orig_to_tok_index, tokenizer)
            if start != -1:
                answer_start.append(start)
                answer_end.append(end)
                answer_text.append(answer)
                is_impossible = False

        if is_impossible:
            answer_start.append(-1)
            answer_end.append(-1)
        else:
            # only take the examples where the answers could be found
            outputs.append((all_doc_tokens, q, answer_start,
                        answer_end, answer_text, is_impossible, sample_id))

    return outputs

def process(s, tokenizer):
    try:
        return tokenizer.tokenize(s)
    except:
        print('failed on', s)
        raise

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def find_ans_span(answer_text, context, char_to_word_offset, doc_tokens, all_doc_tokens, orig_to_tok_index, tokenizer):

    char_start = context.find(answer_text)
    if char_start == -1:
        # print('cannot find the answer')
        return -1, -1

    orig_answer_text = answer_text
    answer_len = len(orig_answer_text)
    start_position = char_to_word_offset[char_start]
    end_position = char_to_word_offset[min(
        char_start + answer_len - 1, len(char_to_word_offset) - 1)]
    tok_start_position = orig_to_tok_index[start_position]

    assert len(orig_to_tok_index) == len(doc_tokens)

    if end_position < len(doc_tokens) - 1:
        tok_end_position = orig_to_tok_index[end_position + 1] - 1
    else:
        tok_end_position = len(all_doc_tokens) - 1

    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
    cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
    if actual_text.find(cleaned_answer_text) == -1:
        print("Could not find answer: '{}' vs. '{}'".format(
            actual_text, cleaned_answer_text))

    (tok_start_position, tok_end_position) = _improve_answer_span(all_doc_tokens, tok_start_position, tok_end_position, process, orig_answer_text, tokenizer)

    return tok_start_position, tok_end_position


def process_files(data_folder, output_folder):
    """
    question:
    answer
    """

    splits = ['train', 'valid']
    tokenizer = BertTokenizer(
        '/private/home/xwhan/fairseq-py/vocab_dicts/vocab.txt', do_lower_case=True)

    for split in splits:

        if not os.path.exists(os.path.join(output_folder, split)):
            os.makedirs(os.path.join(output_folder, split))

        data = open(os.path.join(data_folder, f'{split}.json')).readlines()
        data = [json.loads(_) for _ in data]
        
        num_workers = 50
        chunk_size = len(data) // num_workers
        offsets = [
        _ * chunk_size for _ in range(0, num_workers)] + [len(data)]
        pool = Pool(processes=num_workers)
        print(f'Start multi-processing with {num_workers} workers....')
        results = [pool.apply_async(_process_samples, args=(
            data[offsets[work_id]: offsets[work_id + 1]], tokenizer)) for work_id in range(num_workers)]
        outputs = [p.get() for p in results]
        samples = []
        for o in outputs:
            samples.extend(o)


        question_out = open(os.path.join(output_folder, split, 'q.txt'), 'w')
        context_out = open(os.path.join(output_folder, split, 'c.txt'), 'w')
        answer_start_out = open(os.path.join(
            output_folder, split, 'ans_start.txt'), 'w')
        answer_end_out = open(os.path.join(
            output_folder, split, 'ans_end.txt'), 'w')
        answer_text_out = open(os.path.join(
            output_folder, split, 'ans_text.txt'), 'w')
        sample_id_out = open(os.path.join(
            output_folder, split, 'sample_id.txt'), 'w')

        sample_lens = []
        for sample in samples:
            all_doc_tokens, q, answer_start, answer_end, answer_text, is_impossible, sample_id = sample

            print(' '.join(all_doc_tokens), file=context_out)
            print(' '.join(q), file=question_out)
            print(' '.join([str(ii)
                            for ii in answer_start]), file=answer_start_out)
            print(' '.join([str(ii)
                            for ii in answer_end]), file=answer_end_out)
            print(' '.join([ii for ii in answer_text]), file=answer_text_out)
            print(sample_id, file=sample_id_out)


def binarize_list(words, d):
    return [d.index(w) for w in words]


def debinarize_list(indice, d):
    return [d[int(idx)] for idx in indice]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='WebQ')


    args = parser.parse_args()
    utils.print_args(args)
    process_files(f'/private/home/xwhan/dataset/{args.data}/splits',
                  f'/private/home/xwhan/dataset/{args.data}/processed-splits')



def _improve_answer_span(doc_tokens, input_start, input_end, process,
                         orig_answer_text, tokenizer):
    tok_answer_text = " ".join(process(orig_answer_text, tokenizer))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


if __name__ == '__main__':
    # add_span_and_char_offset()
    main()
