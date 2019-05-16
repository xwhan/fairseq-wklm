#!/usr/bin/env python

try:
    import sweep_chronos
    from sweep_chronos import hyperparam
except:
    import sweep
    from sweep import hyperparam


def get_grid(args):
    return [
        #hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16"),
        hyperparam("--fp16", save_dir_key=lambda val: "me_fp16"),
        #hyperparam("--threshold-loss-scale", 1),
        #hyperparam("--fp16-scale-window", 128),
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
        hyperparam("--valid-subset", "valid"),

        hyperparam("--extra-data",
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard1:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard2:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard3:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard4:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard5:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard6:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard7:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard8:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard9:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard10:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard11:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard12:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard13:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard14:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard15:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard16:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard17:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard18:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard19:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard20:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard21:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard22:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard23:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard24:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard25:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard26:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard27:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard28:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard29:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard30:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard31:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard32:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard33:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard34:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard35:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard36:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard37:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard38:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard39:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard40:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard41:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard42:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard43:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard44:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard45:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard46:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard47:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard48:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard49:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard50:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard51:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard52:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard53:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard54:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard55:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard56:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard57:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard58:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard59:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard60:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard61:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard62:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard63:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard64:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard65:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard66:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard67:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard68:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard69:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard70:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard71:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard72:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard73:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard74:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard75:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard76:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard77:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard78:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard79:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard80:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard81:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard82:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard83:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard84:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard85:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard86:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard87:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard88:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard89:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard90:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard91:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard92:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard93:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard94:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard95:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard96:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard97:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard98:'
        '/private/home/jingfeidu/data/CC-NEWS-en.v11/shard99'),

        #hyperparam("--reset-optimizer", save_dir_key=lambda val: f"resetopt"),

        hyperparam("--task", "bert"),
        hyperparam("--break-mode", "doc"),
        #hyperparam("--add-bos-token", save_dir_key=lambda val: f"bos"),
        #hyperparam("--self-target", save_dir_key=lambda val: f"self"),
        #hyperparam("--word-dropout", 0.15, save_dir_key=lambda val: f"worddrop{val}"),

        #hyperparam("--arch", "bert_hf", save_dir_key=lambda val: val),
        hyperparam("--arch", "bert_hf_large36", save_dir_key=lambda val: val),
        #hyperparam("--arch", "bi_transformer_lm_ccnews_big_16gb", save_dir_key=lambda val: val),
        #hyperparam("--arch", "bi_transformer_lm_ccnews_big", save_dir_key=lambda val: val),
        #hyperparam("--gelu", save_dir_key=lambda val: "gelu"),
        #hyperparam("--share-decoder-input-output-embed", save_dir_key=lambda val: "shareemb"),
        #hyperparam("--decoder-checkpoints", 2, save_dir_key=lambda val: f"deccpt{val}"),

        hyperparam("--criterion", "bert_loss", save_dir_key=lambda val: val),
        #hyperparam("--optimizer", "adafactor", save_dir_key=lambda val: val),
        #hyperparam("--decay-rate", -0.8, save_dir_key=lambda val: f"decay{val}"),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-eps", "1e-08", save_dir_key=lambda val: val),
        hyperparam("--weight-decay", "0.01", save_dir_key=lambda val: val),



        hyperparam("--lr-scheduler", "polynomial_decay"),
        #hyperparam("--warmup-init-lr", 1e-7, save_dir_key=lambda val: f"initlr{val}"),
        #hyperparam("--warmup-updates", 16000, save_dir_key=lambda val: f"warmup{val}"),
        hyperparam("--warmup-updates", 30000, save_dir_key=lambda val: f"warmup{val}"),
        hyperparam("--lr", 1e-4),
        #hyperparam("--max-lr", 1.3e-4, save_dir_key=lambda val: f"lr{val}"),
        #hyperparam("--max-lr", 2e-4, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--max-update", 320000),
        hyperparam("--min-lr", 1e-9),
        hyperparam("--clip-norm", 2, save_dir_key=lambda val: f"clip{val}"),

        #hyperparam("--dropout", 0.0, save_dir_key=lambda val: f"drop{val}"),
        #hyperparam("--attention-dropout", 0.0, save_dir_key=lambda val: f"attndrop{val}"),
        #hyperparam("--weight-decay", 0.0, save_dir_key=lambda val: f"wd{val}"),

        # TODO
        #hyperparam("--update-freq", 4, save_dir_key=lambda val: f"updatefreq{val}"),
        hyperparam("--update-freq", 2, save_dir_key=lambda val: f"updatefreq{val}"),
        hyperparam("--max-tokens", 1024, save_dir_key=lambda val: f"maxtok{val}"),
        hyperparam("--tokens-per-sample", 512, save_dir_key=lambda val: f"sampletok{val}"),
        #hyperparam("--tokens-per-sample", 1024, save_dir_key=lambda val: f"sampletok{val}"),

        hyperparam("--seed", [1], save_dir_key=lambda val: f"seed{val}"),
        hyperparam("--lazy-load", save_dir_key=lambda val: f"lazy-load"),
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
