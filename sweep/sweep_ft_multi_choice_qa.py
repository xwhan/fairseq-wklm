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

        hyperparam('--arch', 'finetuning_sentence_classifier', save_dir_key=lambda val: val),
        hyperparam('--task', 'multi_choice_qa'),

        hyperparam('--max-update', [
            max_update
        ], save_dir_key=lambda val: f'mxup{val}'),
        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--lr', [
           1e-05,2e-05
        ], save_dir_key=lambda val: f'lr{val}'),
        #hyperparam('--t-total', max_update),
        hyperparam('--bert-path', '/checkpoint/myleott/2019-06-01/bookwiki_aml_CC-NEWS-en.v7.1.noactdrop.st512.ms4.mt2200.uf16.masked_lm.no_nsp.bert_large.gelu.dr0.1.atdr0.1.atdr0.0.wd0.01.adam.beta2_98.eps1e-06.clip0.0.lr0.0005.warm10000.me_fp16.mu100000.seed1.ngpu128/checkpoint_18_100000.pt', save_dir_key=lambda val: f'bert'),

        hyperparam('--min-lr', 1e-9),
        hyperparam('--criterion', ['cross_entropy'], save_dir_key=lambda val: f'crs_ent'),
        hyperparam('--sentence-avg', True, binary_flag=True),
        hyperparam('--num-label', 2),
        hyperparam('--max-tokens', [
            1334,
        ], save_dir_key=lambda val: f'mxtk{val}'),

        hyperparam('--seed', [3, 6], save_dir_key=lambda val: f'seed{val}'),

        hyperparam('--skip-invalid-size-inputs-valid-test'),
        hyperparam('--log-format', 'json'),
        hyperparam('--log-interval', [500]),

        hyperparam('--model-dim', 768),
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
