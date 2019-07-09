#!/usr/bin/env python3
# preprocessing code for span style question answering


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


def process_files(data_folder, output_folder):
    """
    question:
    answer
    """
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
            print('cannot find the answer')
            return -1, -1

        orig_answer_text = answer_text
        answer_len = len(orig_answer_text)
        start_position = char_to_word_offset[char_start]
        end_position = char_to_word_offset[min(char_start + answer_len - 1, len(char_to_word_offset) - 1)]
        tok_start_position = orig_to_tok_index[start_position]

        assert len(orig_to_tok_index) == len(doc_tokens)

        if end_position < len(doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            print("Could not find answer: '{}' vs. '{}'".format(actual_text, cleaned_answer_text))

        (tok_start_position, tok_end_position) = _improve_answer_span(all_doc_tokens, tok_start_position, tok_end_position, process, orig_answer_text, tokenizer)

        return tok_start_position, tok_end_position

    def _process_sample(item, tokenizer):

        answer_list = [_.lower() for _ in item['answer']] # multiple answers
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

        orig_to_tok_index = [] # original token to wordpiece index
        tok_to_orig_index = [] # wordpiece token to original token index
        all_doc_tokens = []
        for (i, token) in enumerate(doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = process(token, tokenizer) # wordpiece tokens
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        q = process(item['q'].lower(), tokenizer)

        answer_start = []
        answer_end = []
        answer_text = []
        is_impossible = True
        for answer in answer_list:
            start, end = find_ans_span(answer, context, char_to_word_offset, doc_tokens, all_doc_tokens, orig_to_tok_index, tokenizer)
            if start != -1:
                answer_start.append(start)
                answer_end.append(end)
                answer_text.append(answer)
                is_impossible = False

        if is_impossible:
            answer_start.append(-1)
            answer_end.append(-1)

        return (all_doc_tokens, q, answer_start, answer_end, answer_text, is_impossible, sample_id)

    splits = ['train', 'valid', 'test']
    tokenizer = BertTokenizer('/private/home/xwhan/fairseq-py/vocab_dicts/vocab.txt')

    for split in splits:

        if not os.path.exists(os.path.join(output_folder, split)):
            os.makedirs(os.path.join(output_folder, split))

        data = open(os.path.join(data_folder, f'{split}.json')).readlines()
        data = [json.loads(_) for _ in data]

        samples = Parallel(n_jobs=60)(delayed(_process_sample)(item, tokenizer) for item in tqdm(data))

        question_out = open(os.path.join(output_folder, split, 'q.txt'), 'w')
        context_out = open(os.path.join(output_folder, split, 'c.txt'), 'w')
        answer_start_out = open(os.path.join(output_folder, split, 'ans_start.txt'), 'w')
        answer_end_out = open(os.path.join(output_folder, split, 'ans_end.txt'), 'w')
        answer_text_out = open(os.path.join(output_folder, split, 'ans_text.txt'), 'w')
        sample_id_out = open(os.path.join(output_folder, split, 'sample_id.txt'), 'w')

        for sample in samples:
            all_doc_tokens, q, answer_start, answer_end, answer_text, is_impossible, sample_id = sample

            print(' '.join(all_doc_tokens), file=context_out)
            print(' '.join(q), file=question_out)
            print(' '.join([str(ii) for ii in answer_start]), file=answer_start_out)
            print(' '.join([str(ii) for ii in answer_end]), file=answer_end_out)
            print(' '.join([ii for ii in answer_text]), file=answer_text_out)
            print(sample_id, file=sample_id_out)

        # print(f'Unanswerable questions {num_unanswerable}')

def binarize_list(words, d):
    return [d.index(w) for w in words]

def binarize(args):
    dictionary = BertDictionary.load('/private/home/xwhan/fairseq-py/vocab_dicts/dict.txt')

    splits = ['train', 'valid', 'test']
    for split in splits:
        question_lines = open(os.path.join(args.output, split, 'q.txt')).readlines()
        context_lines = open(os.path.join(args.output, split, 'c.txt')).readlines()
        questions = [binarize_list(line.strip().split(' '), dictionary) for line in question_lines]
        contexts = [binarize_list(line.strip().split(' '), dictionary) for line in context_lines]

        question_out = open(os.path.join(args.output, f'{split}.q'), 'w')
        context_out = open(os.path.join(args.output, f'{split}.c'), 'w')

        assert len(questions) == len(contexts)

        for q, c in zip(questions, contexts):
            print(' '.join([str(ii) for ii in q]), file=question_out)
            print(' '.join([str(ii) for ii in c]), file=context_out)

def debinarize_list(indice, d):
    return [d[int(idx)] for idx in indice]

def debinarize(args):
    dictionary = BertDictionary.load('/private/home/xwhan/fairseq-py/vocab_dicts/dict.txt')

    splits = ['train', 'valid', 'test']
    for split in splits:
        question_lines = open(os.path.join(args.output, f'{split}.q')).readlines()
        context_lines = context_out = open(os.path.join(args.output, f'{split}.c')).readlines()
        questions = [debinarize_list(line.strip().split(' '), dictionary) for line in question_lines]
        contexts = [debinarize_list(line.strip().split(' '), dictionary) for line in context_lines]

        print(questions[:10])
        assert False

        question_out = open(os.path.join(args.output, split, 'q.txt'), 'w')
        context_out = open(os.path.join(args.output, split, 'c.txt'), 'w')

        assert len(questions) == len(contexts)

        for q, c in zip(questions, contexts):
            print(' '.join([str(ii) for ii in q]), file=question_out)
            print(' '.join([str(ii) for ii in c]), file=context_out)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        metavar='DIR',
        help='input split path',
        default='/private/home/xwhan/dataset/webq_qa/splits'
    )
    parser.add_argument(
        '--output',
        metavar='DIR',
        help='Path for output',
        default='/private/home/xwhan/dataset/webq_qa/processed-splits'
    )

    parser.add_argument('--dict_path', default='/private/home/xwhan/fairseq-py/vocab_dicts/dict.txt')
    parser.add_argument('--vocab_path', default='/private/home/xwhan/fairseq-py/vocab_dicts/vocab.txt')

    args = parser.parse_args()
    utils.print_args(args)
    process_files(args.input, args.output)

    # binarize(args)
    # debinarize(args)


def _improve_answer_span(doc_tokens, input_start, input_end, process,
                         orig_answer_text, tokenizer):
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
