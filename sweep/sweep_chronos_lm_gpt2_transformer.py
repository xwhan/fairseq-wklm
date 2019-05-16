#!/usr/bin/env python3

try:
    import sweep_chronos
    from sweep_chronos import hyperparam
except:
    import sweep
    from sweep import hyperparam


def get_grid(args):
    return [
        hyperparam("--lazy-load"),
        hyperparam("--num-workers", 4),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 25),

        hyperparam("--max-update", 100000),
        hyperparam("--save-interval-updates", 4000),

        hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16"),
        #hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        hyperparam("--fp16-scale-window", 128),
        hyperparam("--threshold-loss-scale", 1),

        # TODO can we make c10d work here? There seems to be some
        # incompatibility when using both --memory-efficient-fp16 and
        # --ddp-backend=c10d where we get inconsistent gradients across
        # workers
        hyperparam("--ddp-backend", "no_c10d", save_dir_key=lambda val: "no_c10d"),

        hyperparam("--valid-subset", "valid,valid1"),

        hyperparam("--task", "language_modeling"),
        hyperparam("--sample-break-mode", "none", save_dir_key=lambda val: f"break_{val}"),

        #hyperparam("--arch", "transformer_lm_gpt2_small", save_dir_key=lambda val: val),
        #hyperparam("--arch", "transformer_lm_gpt2_medium", save_dir_key=lambda val: val),
        hyperparam("--arch", "transformer_lm_gpt2_big", save_dir_key=lambda val: val),
        hyperparam("--gelu", save_dir_key=lambda val: "gelu"),
        hyperparam("--decoder-checkpoints", 4, save_dir_key=lambda val: f"deccpt{val}"),
        hyperparam("--share-decoder-input-output-embed", save_dir_key=lambda val: "shareemb"),

        hyperparam("--optimizer", "adafactor", save_dir_key=lambda val: val),
        hyperparam("--decay-rate", -0.8, save_dir_key=lambda val: f"decay{val}"),

        hyperparam("--lr-scheduler", "cosine"),
        hyperparam("--warmup-init-lr", 1e-7, save_dir_key=lambda val: f"initlr{val}"),
        hyperparam("--warmup-updates", 24000, save_dir_key=lambda val: f"warmup{val}"),
        hyperparam("--lr", 1e-4),
        hyperparam("--max-lr", 15e-4, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--max-update", 100000),
        hyperparam("--min-lr", 1e-9),

        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"clip{val}"),
        hyperparam("--dropout", 0.0, save_dir_key=lambda val: f"drop{val}"),
        hyperparam("--weight-decay", 0.0, save_dir_key=lambda val: f"wd{val}"),

        # NOTE use 4 for 128 GPUs and 2 for 256 GPUs
        hyperparam("--update-freq", 4, save_dir_key=lambda val: f"updatefreq{val}"),
        #hyperparam("--update-freq", 2, save_dir_key=lambda val: f"updatefreq{val}"),

        hyperparam("--max-tokens", 1024, save_dir_key=lambda val: f"maxtok{val}"),
        hyperparam("--tokens-per-sample", 1024, save_dir_key=lambda val: f"sampletok{val}"),

        hyperparam("--seed", [1], save_dir_key=lambda val: f"seed{val}"),
    ]

def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == '__main__':
    sweep.main(get_grid, postprocess_hyperparams)
