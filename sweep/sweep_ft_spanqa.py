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

max_update= (100000 // 8) * 15

def get_grid(args):
    return [
        hyperparam('--save-interval', 1),
        # hyperparam('--no-epoch-checkpoints'),
        hyperparam('--arch', 'span_qa', save_dir_key=lambda val: val),
        hyperparam('--task', 'span_qa'),
        hyperparam("--max-epoch", 15),
        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--lr', [1e-5, 5e-6], save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--lr-scheduler', "fixed"),
        hyperparam('--lr-shrink', 0.5),
        # hyperparam('--bert-path', '/checkpoint/xwhan/2019-08-04/kdn_initial_all.adam.bert.crs_ent.seed3.bsz8.0.01.lr0.0001.beta998.warmup10000.ngpu8/checkpoint_1_70000.pt',

        # hyperparam('--bert-path', '/checkpoint/xwhan/2019-08-09/kdn_pred_on_start.adam.bert.crs_ent.seed3.bsz4.0.01.lr1e-05.ngpu16/checkpoint_best.pt',
        #     save_dir_key=lambda val: f'kdn_best'),
        # hyperparam("--use-kdn"),

        hyperparam('--bert-path', '/checkpoint/jingfeidu/2019-05-28/masked-lm-rand.st512.mt4096.uf1.bert_base.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip1.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu32/checkpoint_best.pt',
            save_dir_key=lambda val: f'bert_best'),


        hyperparam('--sentence-avg', True, binary_flag=True),
        hyperparam('--criterion', ['span_qa'], save_dir_key=lambda val: f'crs_ent'),
        hyperparam('--seed', 3, save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--skip-invalid-size-inputs-valid-test'),
        hyperparam('--max-sentences', [8, 16], save_dir_key=lambda val: f'bsz{val}'),
        hyperparam('--log-format', 'json'),
        hyperparam('--log-interval', 1000),
        hyperparam('--model-dim', 768),
        hyperparam("--ddp-backend", "no_c10d"),
        hyperparam('--fp16', True, binary_flag=True),

        # hyperparam('--restore-file', "/checkpoint/xwhan/2019-07-11/reader_squad.span_qa.mxup61875.adam.lr1e-05.bert.crs_ent.seed4.bsz8.ngpu1/checkpoint_best.pt")
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    # if config['--seq-beam'].current_value <= 8:
    #    config['--max-tokens'].current_value = 400
    # else:
    #    config['--max-tokens'].current_value = 300


if __name__ == '__main__':
    sweep.main(get_grid, postprocess_hyperparams)
