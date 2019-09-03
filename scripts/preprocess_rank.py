
import argparse
import json
import os
import sys
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import Pool

sys.path.append('/private/home/xwhan/fairseq-py')

from fairseq.data.masked_lm_dictionary import BertDictionary
from fairseq import utils
from fairseq.tokenization import BertTokenizer, whitespace_tokenize

def _process_samples(items, tokenizer):
    outputs = []
    for item in tqdm(items):
        q_toks = tokenizer.tokenize(item['q'])
        para_toks = tokenizer.tokenize(item['para'])
        label = int(item['label'])
        outputs.append((q_toks, para_toks, label))
    return outputs

def process_files(data_folder, output_folder):
    splits = ['train', 'valid']
    tokenizer = BertTokenizer(
        '/private/home/xwhan/fairseq-py/vocab_dicts/vocab.txt', do_lower_case=True)

    for split in splits:
        if not os.path.exists(os.path.join(output_folder, split)):
            os.makedirs(os.path.join(output_folder, split))
        
        data = open(os.path.join(data_folder, f'{split}.json')).readlines()
        data = [json.loads(_) for _ in data]

        num_workers = 20
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
        lbl_out = open(os.path.join(output_folder, split, 'lbl.txt'), 'w')

        for s in samples:
            print(" ".join(s[0]), file=question_out)
            print(" ".join(s[1]), file=context_out)
            print(str(s[2]), file=lbl_out)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='WebQ')

    args = parser.parse_args()
    utils.print_args(args)
    process_files(f'/private/home/xwhan/DrQA/data/datasets/data/datasets/{args.data}_ranking/splits',
                  f'/private/home/xwhan/DrQA/data/datasets/data/datasets/{args.data}_ranking/processed-splits')


if __name__ == '__main__':
    main()
