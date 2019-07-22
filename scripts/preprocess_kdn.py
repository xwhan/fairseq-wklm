import argparse
import json
import os
import sys
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict

def divide_examples(raw_folder="/private/home/xwhan/Wikipedia/tokenized_debug", outfolder="/private/home/xwhan/dataset/kdn/processed-splits"):
    tokenized_files = os.listdir(raw_folder)

    # read all examples
    samples = []
    for file in tokenized_files:
        file_name = os.path.join(raw_folder, file)
        for line in open(file_name).readlines():
            samples.append(json.loads(line.strip()))

    print(f'Read {len(samples)} samples in total...')

    train_samples = samples[:-500]
    dev_samples = samples[-500:]

    chunk_size = 63
    splits = {'train': train_samples, 'valid': dev_samples}
    num_chunks = 0
    for s, data in splits.items():

        context_out = open(os.path.join(outfolder, s, 'context.txt'), 'w')
        ent_offset_out = open(os.path.join(outfolder, s, 'offset.txt'), 'w')
        ent_len_out = open(os.path.join(outfolder, s, 'len.txt'), 'w')
        ent_lbl_out = open(os.path.join(outfolder, s, 'lbl.txt'), 'w')

        for item in data:
            toks = item['toks']
            ents = item['ents']

            # split article into text blocks
            tok_chunks = []
            for chunk_offset in range(0, len(toks), chunk_size):
                tok_chunks.append(toks[chunk_offset:chunk_offset + chunk_size])
            ent_chunks = defaultdict(list)
            for ent in ents:
                chunk_id = ent["offset"] // chunk_size
                offset_ = ent["offset"] % chunk_size
                if offset_ + len(ent['toks']) > chunk_size: # ignore entities on the boundary
                    continue
                ent_chunks[chunk_id].append({'ent_toks': ent['toks'], 'offset': offset_, 'label': ent['label']})

            num_chunks += len(tok_chunks)

            # write to files
            for idx, tok_chunk in enumerate(tok_chunks):
                co_ents = ent_chunks[idx]
                if len(co_ents) == 0:
                    continue
                offsets = []
                lens = []
                lbls = []
                # check entity offset in each chunk
                for ent in co_ents:
                    offsets.append(ent['offset'])
                    lbls.append(ent['label'])
                    lens.append(len(ent['ent_toks']))
                    assert tok_chunk[offsets[-1]:offsets[-1]+lens[-1]] == ent['ent_toks']

                print(' '.join(tok_chunk), file=context_out)
                print(' '.join([str(ii) for ii in offsets]), file=ent_offset_out)
                print(' '.join([str(ii) for ii in lbls]), file=ent_lbl_out)
                print(' '.join([str(ii) for ii in lens]), file=ent_len_out)

    print(f'Split into {num_chunks} chunks...')

if __name__ == '__main__':
    divide_examples()