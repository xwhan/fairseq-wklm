#!/usr/bin/env python

from functools import reduce
import sweep
from sweep import hyperparam

def get_grid(args):
    return [
        hyperparam('--save-interval', 1),
        hyperparam('--arch', 'span_qa', save_dir_key=lambda val: val),
        hyperparam('--task', 'span_qa'),

        hyperparam("--max-epoch", 20),
        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--lr', [1e-5, 5e-6], save_dir_key=lambda val: f'lr{val}'),
        # hyperparam('--lr', 1e-5,
                #    save_dir_key=lambda val: f'lr{val}'),
        # hyperparam('--lr-scheduler', "reduce_lr_on_plateau"),
        # hyperparam('--lr-shrink', 0.5),
        hyperparam('--final-metric', 'start_acc'),


        # hyperparam('--bert-path', '/checkpoint/xwhan/2019-08-23/kdn_m2_k10_boundary.adam.bert.crs_ent.seed3.bsz4.0.01.lr1e-05.ngpu32/checkpoint_best.pt',
        #            save_dir_key=lambda val: f'kdn_m2_k10'),
        # hyperparam("--use-kdn"),
        # hyperparam('--boundary-loss'),


        # hyperparam('--bert-path', '/checkpoint/jingfeidu/2019-08-23/kdn_v2_boundary_2layer.adam.bert.crs_ent.seed3.bsz4.0.01.lr1e-05.ngpu32/checkpoint_best.pt',
        #            save_dir_key=lambda val: f'kdn_v2_2layer'),
        # hyperparam("--use-kdn"),
        # hyperparam('--boundary-loss'),
        # hyperparam("--add-layer"),
        # hyperparam("--num-kdn", 2),



        # hyperparam('--bert-path', '/checkpoint/xwhan/2019-08-23/kdn_v2_boundary_continue.adam.bert.crs_ent.seed3.bsz4.0.01.lr1e-05.ngpu32/checkpoint_best.pt',
        #            save_dir_key=lambda val: f'kdn_v2_boundary_c'),
        # hyperparam("--use-kdn"),
        # hyperparam('--boundary-loss'),



        # hyperparam('--bert-path', '/checkpoint/xwhan/2019-08-16/kdn_v2_boundary.adam.bert.crs_ent.seed3.bsz4.0.01.lr1e-05.ngpu32/checkpoint_best.pt',
        #            save_dir_key=lambda val: f'kdn_v2_boundary'),
        # hyperparam("--use-kdn"),
        # hyperparam('--boundary-loss'),


        hyperparam('--bert-path', '/checkpoint/jingfeidu/2019-05-28/masked-lm-rand.st512.mt4096.uf1.bert_base.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip1.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu32/checkpoint_best.pt',save_dir_key=lambda val: f'bert_best'),

        # hyperparam('--bert-path', '/checkpoint/xwhan/2019-08-29/kdn_mlm_only.adam.bert.crs_ent.seed3.bsz4.0.01.lr1e-05.ngpu32/checkpoint_best.pt', save_dir_key=lambda val: f'bert_mlm_retrain'),
        # hyperparam("--use-kdn"),
        # hyperparam('--boundary-loss'),

        # # masking ratio 0.05
        # hyperparam('--bert-path', '/checkpoint/xwhan/2019-08-29/kdn_v2_mask0.05.adam.bert.crs_ent.seed3.bsz4.0.01.lr1e-05.ngpu32/checkpoint_best.pt',
        #            save_dir_key=lambda val: f'kdn_v2_mask0.05'),
        # hyperparam("--use-kdn"),
        # hyperparam('--boundary-loss'),

        hyperparam('--sentence-avg', True, binary_flag=True),
        hyperparam('--criterion', 'span_qa'),
        hyperparam('--seed', 3, save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--skip-invalid-size-inputs-valid-test'),
        hyperparam('--max-sentences', 32, save_dir_key=lambda val: f'bsz{val}'),
        hyperparam('--max-length', 128,
                   save_dir_key=lambda val: f'bsz{val}'),

        hyperparam('--log-format', 'json'),
        hyperparam('--log-interval', 1000),
        hyperparam('--model-dim', 768),
        hyperparam("--ddp-backend", "no_c10d"),
        hyperparam('--fp16', True, binary_flag=True),
        hyperparam('--last-dropout', [0.1, 0.2], save_dir_key=lambda val: f'ldrop{val}'),
        # hyperparam('--last-dropout', 0.1,
                #    save_dir_key=lambda val: f'ldrop{val}'),
        # hyperparam('--save-interval-updates', 2000),

        # hyperparam('--restore-file', "/checkpoint/xwhan/2019-08-22/squad_kdn_v2_boundary.span_qa.adam.lr1e-05.kdn_v2_boundary.seed3.bsz8.ldrop0.2.ngpu2/checkpoint_3_16000.pt"),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    # if config['--seq-beam'].current_value <= 8:
    #    config['--max-tokens'].current_value = 400
    # else:
    #    config['--max-tokens'].current_value = 300


if __name__ == '__main__':
    sweep.main(get_grid, postprocess_hyperparams)
