#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Split a large file into shards while respecting document boundaries. Documents
should be separated by a single empty line.
"""

import argparse
import contextlib
import random
import sys
from itertools import zip_longest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='/checkpoint/xwhan/uqa/processed-splits/train/')
    args = parser.parse_args()

    base_dir = args.input_dir
    c_path = base_dir + "/c.txt"
    ans_text_path = base_dir + "/ans_text.txt"
    ans_start_path = base_dir + "/ans_start.txt"
    ans_end_path = base_dir + "/ans_end.txt"
    q_path = base_dir + "/q.txt"
    orig_ans_path = base_dir + "/orig_ans.txt"

    with open(c_path, 'r') as f_c, open(ans_start_path, 'r') as f_start, open(ans_end_path, 'r') as f_end, open(ans_text_path, 'r') as f_ans_text, open(q_path, 'r') as f_q:
        files = [f_c, f_start, f_end, f_ans_text, f_q]
        line_id = 0
        samples = random.sample(range(250484), 50)
        for lines in zip_longest(*files, fillvalue=''):
            starts = lines[1].strip().split()
            ends = lines[2].strip().split()
            ans_texts = lines[3].strip().split('|')
            context = lines[0].strip().split()
            question = lines[4].strip()
            line_id += 1
            # if line_id in [250484, 354416]:
            #     continue
            if line_id in samples:
                print(question)
                print(" ".join(context))
                print(" ".join(ans_texts))
                import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            # assert len(starts) == len(ends) == len(ans_texts)
            # for i in range(len(starts)):
            #     assert context[int(starts[i]):int(ends[i]) + 1] == ans_texts[i].split()
if __name__ == '__main__':
    main()
