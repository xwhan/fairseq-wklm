#!/usr/bin/env python


import sweep
from sweep import hyperparam

def get_grid(args):
    grid = []

    max_update = 1000000
    num_data_loaders = 4

    arch = 'bert_base'
    max_sentences = 8
    update_freq = 1
    peak_lr = 1e-4

    max_tokens = 512 * max_sentences
    ddp_backend = "c10d" if update_freq == 1 else "no_c10d"

    assert (update_freq == 1) ^ (ddp_backend == 'no_c10d')

    def set_data(fmt, num_shards):
        global extra_data
        if num_shards > 0:
            args.data = ":".join([fmt.format(i) for i in range(num_shards)])
        else:
            args.data = fmt

    if args.data == 'CC-NEWS-en.v9':
        set_data('/private/home/namangoyal/fairseq-py/data-bin/CC-NEWS-en.v9/shard{}', 100)
    elif args.data == 'CC-NEWS-en.v9-debug':
        set_data('/private/home/namangoyal/fairseq-py/data-bin/CC-NEWS-en.v9/shard20', 0)
    elif args.data == 'bookwiki':
        set_data('/private/home/yinhanliu/data/bookwiki_aml', 0)
    elif args.data == 'bookwiki_test':
        set_data('/private/home/jingfeidu/data/bert_test', 0)
    elif args.data == 'bookwiki_full':
        set_data('/private/home/myleott/data/data-bin/bookwiki-bin', 0)
    elif args.data == 'fb_posts':
        set_data('/data/tmp/mono.english.public.2018-2019.shard{}.sents.bpe-bin', 100)
    elif args.data == 'fb_posts_gfs':
        set_data('/mnt/vol/gfsai-flash2-east/ai-group/users/myleott/fb_posts/en/mono.english.public.2018-2019.shard{}.sents.bpe-bin', 100)
    elif args.data == 'wmt19_en_news_docs':
        set_data('/private/home/myleott/data/data-bin/wmt19_en_news_docs/wmt19_en_news_docs.bpe.shard{}', 100)
    else:
        raise NotImplementedError

    # batch size
    grid += [
        hyperparam("--tokens-per-sample", 512, save_dir_key=lambda val: f"st{val}"),
       # hyperparam("--max-sentences", max_sentences, save_dir_key=lambda val: f"ms{val}"),
        hyperparam("--max-tokens", max_tokens, save_dir_key=lambda val: f"mt{val}"),
        hyperparam("--update-freq", update_freq, save_dir_key=lambda val: f"uf{val}"),
    ]

    # task settings
    grid += [
        hyperparam("--task", "masked_lm"),
    ]

    # model settings
    grid += [
        hyperparam("--arch", arch, save_dir_key=lambda val: val),
        hyperparam('--criterion', 'masked_lm_loss'),
        hyperparam('--nsp-loss-weight', 1.0),
    ]

    # regularization
    grid += [
        hyperparam("--dropout", 0.1, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--act-dropout", 0.1, save_dir_key=lambda val: f"actdr{val}"),
        hyperparam("--weight-decay", 0.01, save_dir_key=lambda val: f"wd{val}"),
    ]

    # optimization settings
    grid += [
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.999)", save_dir_key=lambda val: "beta998"),
        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"clip{val}"),
        hyperparam("--adam-eps", 1e-6, save_dir_key=lambda val: f"clip{val}"),
    ]

    # lr scheduler
    grid += [
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", peak_lr, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--total-num-update", max_update),
        hyperparam("--warmup-updates", 10000, save_dir_key=lambda val: f"warm{val}"),
        #hyperparam("--reset-lr-scheduler"),
        #hyperparam("--reset-optimizer"),
        #hyperparam("--lr-scheduler", "inverse_sqrt"),
        #hyperparam("--lr", 2e-3, save_dir_key=lambda val: f"lr{val}"),
        #hyperparam("--warmup-init-lr", 0),
        #hyperparam("--warmup-updates", 4000, save_dir_key=lambda val: f"warm{val}"),
    ]

    # FP16 + distributed settings
    grid += [
        hyperparam("--ddp-backend", ddp_backend),

        hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        #hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16"),
        hyperparam("--fp16-init-scale", 128),
        hyperparam("--threshold-loss-scale", 1),
    ]

    # data loading settings
    grid += [
        #hyperparam("--lazy-load"),
        hyperparam("--num-workers", num_data_loaders),
    ]

    # validation and checkpoint settings
    grid += [
        hyperparam("--save-interval-updates", 20000),
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--max-update", max_update, save_dir_key=lambda val: f"mu{val}"),
    ]

    # logging settings
    grid += [
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 100),
    ]

    # random seed
    grid += [
        hyperparam("--seed", [1], save_dir_key=lambda val: f"seed{val}"),
    ]

    if args.local:
        grid += [
            hyperparam("--log-format", "json"),
            hyperparam("--log-interval", 1),
        ]

    return grid


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == '__main__':
    sweep.main(get_grid, postprocess_hyperparams)
