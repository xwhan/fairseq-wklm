#!/usr/bin/env python

from functools import reduce
import sweep
from sweep import hyperparam



max_update = 100000

def get_grid(args):
    return [

        hyperparam('--save-interval', 1),
        hyperparam('--arch', 'finetuning_paragraph_ranker', save_dir_key=lambda val: val),
        hyperparam('--task', 'paragaph_ranking'),

        hyperparam("--max-epoch", 5),
        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--lr', 1e-5, save_dir_key=lambda val: f'lr{val}'),

        hyperparam('--bert-path', '/checkpoint/jingfeidu/2019-05-28/masked-lm-rand.st512.mt4096.uf1.bert_base.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip1.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu32/checkpoint_best.pt', save_dir_key=lambda val: f'bert'),

        hyperparam('--criterion', ['cross_entropy'], save_dir_key=lambda val: f'crs_ent'),
        hyperparam('--sentence-avg', True, binary_flag=True),
        hyperparam('--num-label', 2),
        hyperparam('--seed', 3, save_dir_key=lambda val: f'seed{val}'),

        hyperparam('--skip-invalid-size-inputs-valid-test'),
        hyperparam('--log-format', 'json'),
        hyperparam('--log-interval', 1000),
        hyperparam('--max-sentences', 4, save_dir_key=lambda val: f'bsz{val}'),

        hyperparam('--model-dim', 768),
        hyperparam("--ddp-backend", "no_c10d"),
        hyperparam('--fp16', True, binary_flag=True),
        hyperparam('--last-dropout', [0.1, 0.2], save_dir_key=lambda val: f'ldrop{val}'),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    # if config['--seq-beam'].current_value <= 8:
    #    config['--max-tokens'].current_value = 400
    # else:
    #    config['--max-tokens'].current_value = 300


if __name__ == '__main__':
    sweep.main(get_grid, postprocess_hyperparams)
