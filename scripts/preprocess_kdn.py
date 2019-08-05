import argparse
import json
import os
from tqdm import tqdm
from collections import defaultdict
import itertools

from multiprocessing import Pool

"""
divide article data into text chunks
input format:
line 1: space-separated wordpiece toks
line 2: offset len lbl||offset len lbl
line 3: entities' wordpiece toks
"""

def read_file(file_path):
    toks = []
    ent_infos = []
    ent_toks = []

    def read_ents(line):
        ent_triples = [item.split(" ") for item in line.strip().split('||')]
        try:
            ents_ = [{"offset":int(item[0]), "len": int(item[1]), "label": int(item[2])} for item in ent_triples]
        except:
            return []
        
        return ents_

    def read_ent_toks(line):
        ent_toks = []
        ent_spans = line.strip().split("||")
        for span in ent_spans:
            ent_toks.append(span.split(" "))
        return ent_toks

    with open(file_path) as f:
        lines = f.readlines()
    print(f'load data from {file_path}')
    assert len(lines) % 3 == 0
    for line_idx, line in enumerate(tqdm(lines)):
        if line_idx % 3 == 0:
            toks.append(line.strip().split(" "))
        elif line_idx % 3 == 1:
            ent_infos.append(read_ents(line))
        else:
            # ent_toks.append(read_ent_toks(line))
            continue

    assert len(ent_infos) == len(toks)
    return list(zip(toks, ent_infos))


def divide_examples(raw_folder="/private/home/xwhan/Wikipedia/tokenized", outfolder="/private/home/xwhan/dataset/kdn/processed-splits", chunk_size=512, num_workers=10):
    tokenized_files = os.listdir(raw_folder)
    chunk_size = chunk_size - 1

    print(f'how many files {len(tokenized_files)}......')

    # for k in itertools.count():
    for k in [99]:
        if k == len(tokenized_files):
            return
        file = os.path.join(raw_folder, tokenized_files[k])
        samples = read_file(file) # tokens_and_ents tuples
        print(f'Read {len(samples)} samples in total from {k}st file...')

        if k == len(tokenized_files) - 1: # the last file
            valid_num = int(0.1 * len(samples))
            splits = {'train': samples[:-valid_num], 'valid': samples[-valid_num:]}
        else:
            splits = {'train': samples}

        num_chunks = 0
        for s, data in splits.items():
            if s == 'valid':
                k = 0

            context_out = open(os.path.join(outfolder, s, f'context_{k}.txt'), 'w')
            ent_offset_out = open(os.path.join(outfolder, s, f'offset_{k}.txt'), 'w')
            ent_len_out = open(os.path.join(outfolder, s, f'len_{k}.txt'), 'w')
            ent_lbl_out = open(os.path.join(outfolder, s, f'lbl_{k}.txt'), 'w')

            for item in data:
                toks = item[0]
                ents = item[1]

                # split article into text blocks
                tok_chunks = []
                for chunk_offset in range(0, len(toks), chunk_size):
                    tok_chunks.append(toks[chunk_offset:chunk_offset + chunk_size])
                ent_chunks = defaultdict(list)
                for ent in ents:
                    chunk_id = ent["offset"] // chunk_size
                    offset_ = ent["offset"] % chunk_size
                    if offset_ + ent['len'] > chunk_size: # ignore entities on the boundary
                        continue
                    ent_chunks[chunk_id].append({'ent_len': ent['len'], 'offset': offset_, 'label': ent['label']})
                num_chunks += len(tok_chunks)

                # write to files
                for idx, tok_chunk in enumerate(tok_chunks):
                    co_ents = ent_chunks[idx]
                    if len(co_ents) == 0:
                        continue
                    offsets = []
                    lens = []
                    lbls = []

                    for ent in co_ents:
                        if ent['ent_len'] == 0:
                            continue
                        offsets.append(ent['offset'])
                        lbls.append(ent['label'])
                        lens.append(ent['ent_len'])

                    print(' '.join(tok_chunk), file=context_out)
                    print(' '.join([str(ii) for ii in offsets]), file=ent_offset_out)
                    print(' '.join([str(ii) for ii in lbls]), file=ent_lbl_out)
                    print(' '.join([str(ii) for ii in lens]), file=ent_len_out)

            context_out.close()
            ent_offset_out.close()
            ent_len_out.close()
            ent_lbl_out.close()

        del samples, splits

        print(f'Split into {num_chunks} chunks...')

if __name__ == '__main__':
    divide_examples(raw_folder="/private/home/xwhan/Wikipedia/tokenized", outfolder="/checkpoint/xwhan/wiki_data/processed-splits")
