#!/usr/bin/env python

from functools import reduce
import sys
import sweep
from sweep import hyperparam


def get_lr_str(val):
    uniq_lr = reduce(
        lambda x, y: x + y if x[-1] != y[-1] else x,
        map(lambda x: [x], val.split(','))
    )
    return ','.join(uniq_lr)


def get_filter_str(val):
    f = eval(val)
    s = f'cf{f[0][1]}'
    return s


max_update = 33135*4

def get_grid(args):
    return [

        hyperparam('--save-interval', 1),
        hyperparam('--no-epoch-checkpoints'),
        #hyperparam('--warmup', 0.1),

        hyperparam('--arch', 'finetuning_sentence_pair_classifier', save_dir_key=lambda val: val),
        hyperparam('--task', 'sentence_pair_classification'),

        hyperparam('--max-update', [
            max_update
        ], save_dir_key=lambda val: f'mxup{val}'),
        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--lr', [
           1e-05,2e-05
        ], save_dir_key=lambda val: f'lr{val}'),
        #hyperparam('--t-total', max_update),
        hyperparam('--bert-path', '/checkpoint/ves/2019-06-02/mlm-big-ccnews.st512.mt2048.uf1.bert_large.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip4.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu64/checkpoint_best.pt',
            save_dir_key=lambda val: f'bert'),

        hyperparam('--min-lr', 1e-9),
        hyperparam('--criterion', ['cross_entropy'], save_dir_key=lambda val: f'crs_ent'),
        hyperparam('--sentence-avg', True, binary_flag=True),
        hyperparam('--num-labels', 3),
        hyperparam('--max-tokens', [
            1334,
        ], save_dir_key=lambda val: f'mxtk{val}'),

        hyperparam('--seed', [3, 6], save_dir_key=lambda val: f'seed{val}'),

        hyperparam('--skip-invalid-size-inputs-valid-test'),
        hyperparam('--log-format', 'json'),
        hyperparam('--log-interval', [500]),

        hyperparam('--model-dim', 1024),
        #hyperparam('--mnli-dropout', [0.1], save_dir_key=lambda val: f'f_drp{val}'),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    # if config['--seq-beam'].current_value <= 8:
    #    config['--max-tokens'].current_value = 400
    # else:
    #    config['--max-tokens'].current_value = 300


if __name__ == '__main__':
    sweep.main(get_grid, postprocess_hyperparams)
