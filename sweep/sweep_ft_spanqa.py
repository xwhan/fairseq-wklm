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
        hyperparam('--arch', 'span_qa', save_dir_key=lambda val: val),
        hyperparam('--task', 'span_qa'),
        hyperparam("--max-epoch", 8),
        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--lr', [1e-5, 2e-5, 5e-5], save_dir_key=lambda val: f'lr{val}'),
        # hyperparam('--lr', 1e-5,
                #    save_dir_key=lambda val: f'lr{val}'),
        # hyperparam('--lr-scheduler', "reduce_lr_on_plateau"),
        # hyperparam('--lr-shrink', 0.5),
        hyperparam('--final-metric', 'start_acc'),

        # hyperparam('--bert-path', '/checkpoint/xwhan/2019-08-16/kdn_v3_start_end.adam.bert.crs_ent.seed3.bsz4.0.01.lr1e-05.ngpu32/checkpoint_best.pt',
        #            save_dir_key=lambda val: f'kdn_best'),
        # hyperparam("--use-kdn"),
        # hyperparam("--start-end"),

        hyperparam('--bert-path', '/checkpoint/xwhan/2019-08-16/kdn_v3_start_add_4_layer.adam.bert.crs_ent.seed3.bsz4.0.01.lr1e-05.ngpu32/checkpoint_best.pt',
                   save_dir_key=lambda val: f'kdn_v3_start_add4'),
        hyperparam("--use-kdn"),
        hyperparam('--add-layer'),

        # hyperparam('--bert-path', '/checkpoint/jingfeidu/2019-05-28/masked-lm-rand.st512.mt4096.uf1.bert_base.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip1.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu32/checkpoint_best.pt',save_dir_key=lambda val: f'bert_best'),


        hyperparam('--sentence-avg', True, binary_flag=True),
        hyperparam('--criterion', ['span_qa'], save_dir_key=lambda val: f'crs_ent'),
        hyperparam('--seed', 3, save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--skip-invalid-size-inputs-valid-test'),
        hyperparam('--max-sentences', 8, save_dir_key=lambda val: f'bsz{val}'),
        hyperparam('--log-format', 'json'),
        hyperparam('--log-interval', 1000),
        hyperparam('--model-dim', 768),
        hyperparam("--ddp-backend", "no_c10d"),
        hyperparam('--fp16', True, binary_flag=True),
        hyperparam('--last-dropout', [0.1, 0.2], save_dir_key=lambda val: f'ldrop{val}'),
        # hyperparam('--use-shards'),

        # hyperparam('--restore-file', "/checkpoint/xwhan/2019-08-20/uqa_bert.span_qa.adam.lr1e-05.bert_best.crs_ent.seed3.bsz8.ldrop0.1.ngpu32/checkpoint_best.pt")
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    # if config['--seq-beam'].current_value <= 8:
    #    config['--max-tokens'].current_value = 400
    # else:
    #    config['--max-tokens'].current_value = 300


if __name__ == '__main__':
    sweep.main(get_grid, postprocess_hyperparams)
