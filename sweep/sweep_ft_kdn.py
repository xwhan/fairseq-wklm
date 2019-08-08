#!/usr/bin/env python

from functools import reduce
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

def get_grid(args):
    return [
        hyperparam('--save-interval', 1),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--arch', 'kdn'),
        hyperparam('--task', 'kdn'),
        # hyperparam('--max-epoch', 5),
        hyperparam("--max-update", 1000000),
        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
                hyperparam('--bert-path', '/checkpoint/jingfeidu/2019-05-28/masked-lm-rand.st512.mt4096.uf1.bert_base.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip1.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu32/checkpoint_best.pt',
            save_dir_key=lambda val: f'bert'),
        hyperparam('--criterion', 'kdn_loss', save_dir_key=lambda val: f'crs_ent'),
        hyperparam('--log-format', 'json'),
        hyperparam('--seed', 3, save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--max-sentences', 8, save_dir_key=lambda val: f'bsz{val}'),
        # hyperparam("--weight-decay", "0.01", save_dir_key=lambda val: val),
        hyperparam('--lr', 1e-5, save_dir_key=lambda val: f'lr{val}'),
        # hyperparam("--adam-betas", "(0.9, 0.999)", save_dir_key=lambda val: "beta998"),
        hyperparam("--lr-scheduler", "fixed"),
        hyperparam('--save-interval-updates', 5000),
        # hyperparam("--warmup-updates", 10000, save_dir_key=lambda val: f"warmup{val}"),
        hyperparam("--tensorboard-logdir", "/checkpoint/xwhan/kdn_pretrain")
        hyperparam('--log-interval', 1000),
        hyperparam('--model-dim', 768),
        hyperparam('--fp16', True, binary_flag=True),
        hyperparam('--use-mlm'),
        hyperparam('--ddp-backend', "no_c10d"),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    # if config['--seq-beam'].current_value <= 8:
    #    config['--max-tokens'].current_value = 400
    # else:
    #    config['--max-tokens'].current_value = 300


if __name__ == '__main__':
    sweep.main(get_grid, postprocess_hyperparams)
