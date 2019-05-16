#!/usr/bin/env python

try:
    import sweep_chronos
    from sweep_chronos import hyperparam
except:
    import sweep
    from sweep import hyperparam


def get_grid(args):
    return [
        hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16"),
        #hyperparam("--fp16", save_dir_key=lambda val: "me_fp16"),
        hyperparam("--threshold-loss-scale", 1),
        hyperparam("--fp16-scale-window", 128),
        #hyperparam("--ddp-backend", "no_c10d", save_dir_key=lambda val: "no_c10d"),
        #hyperparam("--lazy-load"),
        #hyperparam("--num-workers", 4),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 25),
        hyperparam("--max-update", 320000),
        hyperparam("--save-interval-updates", 4000),

        # TODO
        #hyperparam("--valid-subset", "valid,valid1"),
        #hyperparam("--train-subset", "valid"),
        #hyperparam("--valid-subset", "valid"),

        #hyperparam("--reset-optimizer", save_dir_key=lambda val: f"resetopt"),

        hyperparam("--task", "language_modeling"),
        hyperparam("--sample-break-mode", "none"),
        #hyperparam("--add-bos-token", save_dir_key=lambda val: f"bos"),
        hyperparam("--self-target", save_dir_key=lambda val: f"self"),
        #hyperparam("--word-dropout", 0.15, save_dir_key=lambda val: f"worddrop{val}"),

        hyperparam("--arch", "bi_transformer_lm_big_32gb", save_dir_key=lambda val: val),
        #hyperparam("--arch", "bi_transformer_lm_ccnews_big_16gb", save_dir_key=lambda val: val),
        #hyperparam("--arch", "bi_transformer_lm_ccnews_big", save_dir_key=lambda val: val),
        hyperparam("--gelu", save_dir_key=lambda val: "gelu"),
        hyperparam("--share-decoder-input-output-embed", save_dir_key=lambda val: "shareemb"),
        #hyperparam("--decoder-checkpoints", 2, save_dir_key=lambda val: f"deccpt{val}"),

        hyperparam("--optimizer", "adafactor", save_dir_key=lambda val: val),
        hyperparam("--decay-rate", -0.8, save_dir_key=lambda val: f"decay{val}"),

        hyperparam("--lr-scheduler", "cosine"),
        hyperparam("--warmup-init-lr", 1e-7, save_dir_key=lambda val: f"initlr{val}"),
        #hyperparam("--warmup-updates", 16000, save_dir_key=lambda val: f"warmup{val}"),
        hyperparam("--warmup-updates", 10400, save_dir_key=lambda val: f"warmup{val}"),
        hyperparam("--lr", 1e-4),
        #hyperparam("--max-lr", 1.3e-4, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--max-lr", 2e-4, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--max-update", 320000),
        hyperparam("--min-lr", 1e-9),
        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"clip{val}"),

        hyperparam("--dropout", 0.0, save_dir_key=lambda val: f"drop{val}"),
        hyperparam("--attention-dropout", 0.0, save_dir_key=lambda val: f"attndrop{val}"),
        hyperparam("--weight-decay", 0.0, save_dir_key=lambda val: f"wd{val}"),

        # TODO
        #hyperparam("--update-freq", 4, save_dir_key=lambda val: f"updatefreq{val}"),
        hyperparam("--update-freq", 2, save_dir_key=lambda val: f"updatefreq{val}"),
        hyperparam("--max-tokens", 1024, save_dir_key=lambda val: f"maxtok{val}"),
        hyperparam("--tokens-per-sample", 512, save_dir_key=lambda val: f"sampletok{val}"),
        #hyperparam("--tokens-per-sample", 1024, save_dir_key=lambda val: f"sampletok{val}"),

        hyperparam("--seed", [1], save_dir_key=lambda val: f"seed{val}"),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    # if config['--seq-beam'].current_value <= 8:
    #    config['--max-tokens'].current_value = 400
    # else:
    #    config['--max-tokens'].current_value = 300
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
