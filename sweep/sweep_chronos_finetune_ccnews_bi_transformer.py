#!/usr/bin/env python

try:
    import sweep_chronos
    from sweep_chronos import hyperparam
except:
    import sweep
    from sweep import hyperparam


def get_grid(args):
    return [
        # NOTE for debugging
        #hyperparam("--train-subset", "valid"),

        hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16"),
        #hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        hyperparam("--threshold-loss-scale", 1),
        hyperparam("--fp16-scale-window", 128),
        hyperparam("--ddp-backend", "no_c10d", save_dir_key=lambda val: "no_c10d"),
        #hyperparam("--lazy-load"),
        #hyperparam("--num-workers", 4),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 25),

        # this will cause validation every 250 updates, but will not save checkpoints
        hyperparam("--no-save"),
        hyperparam("--save-interval-updates", 250),

        hyperparam("--task", "finetune_lm_glue"),
        hyperparam("--model-output-dim", 1536, save_dir_key=lambda val: f"dim{val}"),
        hyperparam("--scale-factor", [0.1, 0.3, 1], save_dir_key=lambda val: f"scale{val}"),
        hyperparam("--final-dropout", [0.2, 0.3, 0.4, 0.5], save_dir_key=lambda val: f"finaldrop{val}"),

        hyperparam("--arch", "bi_transformer_lm_ccnews_big", save_dir_key=lambda val: val),
        hyperparam("--share-decoder-input-output-embed", save_dir_key=lambda val: "shareemb"),
        hyperparam("--unmask-curr-state", save_dir_key=lambda val: "unmask"),
        hyperparam("--remove-head"),

        # TODO
        hyperparam("--remove-last", [2], save_dir_key=lambda val: f"rm{val}"),
        #hyperparam("--only-endpoints", save_dir_key=lambda val: f"ends"),
        hyperparam("--left-right-selfattn", save_dir_key=lambda val: f"lrselfattn"),

        hyperparam("--criterion", "glue", save_dir_key=lambda val: val),

        # TODO try not resetting optimizer (it doesn't work currently)
        hyperparam("--reset-optimizer", save_dir_key=lambda val: f"resetopt"),
        #hyperparam("--force-reuse-optimizer", save_dir_key=lambda val: f"reuseopt"),

        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "beta0.9,0.98"),
        hyperparam("--lr-scheduler", "fixed"),
        #hyperparam("--lr-scheduler", "inverse_sqrt"),
        #hyperparam("--warmup-init-lr", 1e-7, save_dir_key=lambda val: f"initlr{val}"),
        #hyperparam("--warmup-updates", 4000, save_dir_key=lambda val: f"warmup{val}"),
        hyperparam("--min-lr", 1e-9),
        hyperparam("--lr", [1e-5, 3e-5, 1e-4], save_dir_key=lambda val: f"lr{val}"),

        #hyperparam("--optimizer", "adafactor", save_dir_key=lambda val: val),
        #hyperparam("--decay-rate", -0.8, save_dir_key=lambda val: f"decay{val}"),
        #hyperparam("--lr-scheduler", "cosine"),
        #hyperparam("--warmup-init-lr", 1e-7, save_dir_key=lambda val: f"initlr{val}"),
        #hyperparam("--warmup-updates", 4000, save_dir_key=lambda val: f"warmup{val}"),
        #hyperparam("--lr", 1e-5),
        #hyperparam("--max-lr", [1e-5, 3e-5, 1e-4, 3e-4], save_dir_key=lambda val: f"lr{val}"),
        #hyperparam("--max-update", 20000),
        #hyperparam("--min-lr", 1e-9),

        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"clip{val}"),
        #hyperparam("--dropout", [0.0, 0.1], save_dir_key=lambda val: f"drop{val}"),
        hyperparam("--dropout", [0.0], save_dir_key=lambda val: f"drop{val}"),
        hyperparam("--attention-dropout", 0.0, save_dir_key=lambda val: f"attndrop{val}"),
        hyperparam("--weight-decay", 0.0, save_dir_key=lambda val: f"wd{val}"),

        # TODO
        #hyperparam("--update-freq", [1, 2, 4], save_dir_key=lambda val: f"updatefreq{val}"),
        hyperparam("--max-sentences", [32], save_dir_key=lambda val: f"bsz{val}"),
        #hyperparam("--max-tokens", 1024, save_dir_key=lambda val: f"maxtok{val}"),
        #hyperparam("--tokens-per-sample", 1024, save_dir_key=lambda val: f"sampletok{val}"),

        hyperparam("--seed", [1], save_dir_key=lambda val: f"s{val}"),
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
