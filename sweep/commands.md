# Paragraph Ranker Experiemnts
## debug para ranker
python train.py --fp16  /private/home/xwhan/dataset/WebQ-Ranking --task paragaph_ranking --arch finetuning_paragraph_ranker --save-interval 1 --max-update 30000 --lr 2e-05 --bert-path /checkpoint/jingfeidu/2019-05-28/masked-lm-rand.st512.mt4096.uf1.bert_base.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip1.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu32/checkpoint_best.pt --num-label 2 --distributed-world-size 1 --max-sentences 8 --optimizer adam 
## sweep for paragraph ranking
python sweep/sweep_ft_ranker.py -d /private/home/xwhan/dataset/WebQ-Ranking -p WebQ_ranking_baseline -t -1 -g 1 -n 1 --tensorboard-logdir /checkpoint/xwhan/ranking 
## Evaluation 
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_ranker.py /private/home/xwhan/dataset/WebQ --model-path /checkpoint/xwhan/2019-08-18/WebQ_ranking_baseline.finetuning_paragraph_ranker.adam.lr1e-05.bert.crs_ent.seed3.bsz8.ldrop0.1.ngpu1/checkpoint_last.pt -a finetuning_paragraph_ranker

------------------------------------

# Span QA Experiments
## Baseline Debug
python train.py --fp16 /checkpoint/xwhan/uqa --task span_qa --arch span_qa --save-interval 1 --max-update 30000 --lr 1e-05 --bert-path /checkpoint/jingfeidu/2019-05-28/masked-lm-rand.st512.mt4096.uf1.bert_base.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip1.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu32/checkpoint_best.pt --distributed-world-size 1 --max-sentences 8 --optimizer adam --criterion span_qa --final-metric start_acc --last-dropout 0.1 --use-shards --save-interval-updates 50 
## KDN Debug
python train.py --fp16 /private/home/xwhan/dataset/webq_qa --task span_qa --arch span_qa --save-interval 1 --max-update 30000 --lr 1e-05 --bert-path /checkpoint/xwhan/2019-08-16/kdn_v3_start_add_4_layer.adam.bert.crs_ent.seed3.bsz4.0.01.lr1e-05.ngpu32/checkpoint_best.pt --distributed-world-size 1 --max-sentences 8 --optimizer adam --criterion span_qa --save-interval-updates 10 --final-metric start_acc --use-kdn --add-layer


## sweep for WebQuesions
* WebQ bert baseline
python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/dataset/triviaqa -p triviaqa_baseline -t -1 -g 2 -n 1

* WebQ kdn pred on starts v2
python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/dataset/WebQ -p WebQ_kdn -t -1 -g 1 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa



## sweep for SQuAD 1.1 
python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/dataset/squad1.1 -p squad_kdn_v3_start_end -t -1 -g 2 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa

python sweep/sweep_ft_spanqa.py -d /checkpoint/xwhan/uqa -p uqa_only_first_entity -t -1 -g 4 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa

## evaluation for WebQ
* use kdn model 
```
python scripts/evaluate_reader.py /private/home/xwhan/dataset/webq_qa --use-kdn --model-path /checkpoint/xwhan/2019-08-16/webq_kdn.span_qa.adam.lr1e-05.kdn_best.crs_ent.seed3.bsz8.ngpu1/checkpoint_last.pt --start-end -a span_qa
```
* use bert model
```
python scripts/evaluate_reader.py /private/home/xwhan/dataset/webq_qa --model-path /checkpoint/xwhan/2019-08-20/uqa_bert.span_qa.adam.lr1e-05.bert_best.crs_ent.seed3.bsz8.ldrop0.1.ngpu32/checkpoint_best.pt --arch span_qa 
```

## Evaluation for SQuAD
* KDN model
```
python scripts/evaluate_reader.py /private/home/xwhan/dataset/squad1.1 --model-path /checkpoint/xwhan/2019-08-20/uqa_bert_rerun.span_qa.adam.lr1e-05.bert_best.crs_ent.seed3.bsz8.ldrop0.1.ngpu32/checkpoint_last.pt --arch span_qa --eval-data /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json --answer-path /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json
```
* BERT model
python scripts/evaluate_reader.py /private/home/xwhan/dataset/squad1.1 --model-path /checkpoint/xwhan/2019-08-18/squad_bert.span_qa.adam.lr2e-05.bert_best.crs_ent.seed3.bsz8.ldrop0.2.ngpu2/checkpoint_best.pt --arch span_qa --eval-data /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json --answer-path /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json

------------------------------------

# Cancel all jobs
squeue -u xwhan | grep 1695 | awk '{print $1}' | xargs -n 1 scancel

------------------------------------
# KDN Pretrainning Experiments
# kdn debug
python train.py --fp16 /checkpoint/xwhan/wiki_data --task kdn --arch kdn --save-interval 1 --max-update 1000000 --lr 1e-05 --bert-path /checkpoint/jingfeidu/2019-05-28/masked-lm-rand.st512.mt4096.uf1.bert_base.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip1.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu32/checkpoint_best.pt --distributed-world-size 1 --max-sentences 4 --optimizer adam --criterion kdn_loss --ddp-backend no_c10d --save-interval-updates 1000 --use-mlm --last-dropout 0.1 --start-end --add-layer --num-kdn 6 --masking_ratio
# sweep for kdn
python sweep/sweep_ft_kdn.py -d /checkpoint/xwhan/wiki_data -p kdn_v3_start_add_6_layer -t -1 -g 8 -n 4

# data processing flow
* replace the entities, in process_wiki, `python process_wikipedia.py`
* split into 512 chunks, in fairseq-py, `python scripts/preprocess_kdn.py`
* binarize the data with sweep

------------------------------------
# Relation Extraction Experiments

## debug relation extraction
/checkpoint/ves/2019-05-31/mlm-big-bookwiki.st512.mt4096.uf1.bert_large.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip4.0.adam_eps6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu64/checkpoint_best.pt

/checkpoint/jingfeidu/2019-05-28/masked-lm-rand.st512.mt4096.uf1.bert_base.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip1.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu32/checkpoint_best.pt

python train.py --fp16  /private/home/xwhan/dataset/tacred --task re --arch re --save-interval 1 --max-update 30000 --lr 2e-05 --bert-path /checkpoint/xwhan/2019-08-16/kdn_v2_boundary.adam.bert.crs_ent.seed3.bsz4.0.01.lr1e-05.ngpu32/checkpoint_best.pt --distributed-world-size 1 --max-sentences 8 --optimizer adam --criterion cross_entropy --use-marker --curriculum 1 --ddp-backend no_c10d --save-interval-updates 10 --final-metric f1 --use-kdn --boundary-loss

```use
python train.py --fp16  /private/home/xwhan/dataset/tacred --task re --arch re --save-interval 1 --max-update 30000 --lr 2e-05 --bert-path /checkpoint/ves/2019-05-31/mlm-big-bookwiki.st512.mt4096.uf1.bert_large.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip4.0.adam_eps6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu64/checkpoint_best.pt --distributed-world-size 1 --max-sentences 8 --optimizer adam --criterion cross_entropy --use-marker --curriculum 1 --model-dim 1024 --ddp-backend no_c10d --use-hf --use-cased
```

## evaluate relation extraction
Use BERT model
```
python scripts/evaluate_re.py --arch re /private/home/xwhan/dataset/tacred --model-path /checkpoint/xwhan/2019-08-15/re_base_uncased_marker.re.adam.lr2e-05.bert.crs_ent.seed3.bsz32.maxlen128.drop0.1.ngpu1/checkpoint_last.ptvxa
```

## sweep relation extraction
python sweep/sweep_ft_re.py -d /private/home/xwhan/dataset/tacred -p re_kdn_marker -t -1 -g 2 -n 1 

# tensorboard logs
ssh -J prn-fairjmp02 -L 8889:localhost:8889 100.97.67.36


