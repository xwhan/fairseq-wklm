# Paragraph Ranker Experiemnts
## debug para ranker
python train.py --fp16  /private/home/xwhan/dataset/triviaqa-ranking --task paragaph_ranking --arch finetuning_paragraph_ranker --save-interval 1 --max-update 30000 --lr 2e-05 --bert-path /checkpoint/jingfeidu/2019-05-28/masked-lm-rand.st512.mt4096.uf1.bert_base.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip1.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu32/checkpoint_best.pt --num-label 2 --distributed-world-size 2 --max-sentences 8 --optimizer adam --ddp-backend no_c10d
## sweep for paragraph ranking
python sweep/sweep_ft_ranker.py -d /private/home/xwhan/dataset/triviaqa-ranking -p triviaqa_ranking_baseline -t -1 -g 8 -n 1 --tensorboard-logdir /checkpoint/xwhan/ranking 
## Evaluation 
python scripts/evaluate_ranker.py /private/home/xwhan/dataset/triviaqa --model-path /checkpoint/xwhan/2019-08-24/triviaqa_ranking_baseline.finetuning_paragraph_ranker.adam.lr1e-05.bert.crs_ent.seed3.bsz4.ldrop0.2.ngpu8/checkpoint_best.pt -a finetuning_paragraph_ranker

------------------------------------

# Span QA Experiments
## Baseline Debug
python train.py --fp16 /private/home/xwhan/dataset/triviaqa --task span_qa --arch span_qa --save-interval 1 --max-update 30000 --lr 1e-05 --bert-path /checkpoint/jingfeidu/2019-05-28/masked-lm-rand.st512.mt4096.uf1.bert_base.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip1.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu32/checkpoint_best.pt --distributed-world-size 1 --max-sentences 8 --optimizer adam --criterion span_qa --final-metric start_acc --last-dropout 0.05  --save-interval-updates 50 


## KDN Debug
python train.py --fp16 /private/home/xwhan/dataset/squad1.1 --task span_qa --arch span_qa --save-interval 1 --max-update 30000 --lr 1e-05 --bert-path /checkpoint/jingfeidu/2019-08-23/kdn_v2_boundary_2layer.adam.bert.crs_ent.seed3.bsz4.0.01.lr1e-05.ngpu32/checkpoint_best.pt --distributed-world-size 1 --max-sentences 8 --optimizer adam --criterion span_qa --save-interval-updates 10 --final-metric start_acc --use-kdn --boundary-loss -add-layer --num-kdn 2


## BCELoss debug


## sweep for WebQuesions
* WebQ bert baseline
python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/dataset/WebQ -p WebQ_mlm_ablation -t -1 -g 1 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa

python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/dataset/WebQ -p WebQ_kdn_m0.05 -t -1 -g 1 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa

python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/dataset/WebQ -p WebQ_ablation_1m -t -1 -g 1 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa

python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/dataset/triviaqa -p triviaqa_kdn_filtered_b16 -t -1 -g 2 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa

python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/dataset/WebQ -p WebQ_kdn -t -1 -g 1 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa

* WebQ kdn pred on starts v2
python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/dataset/WebQ -p WebQ_kdn_v2_boundary_continue -t -1 -g 1 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa


## sweep for SQuAD 1.1 
python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/dataset/squad1.1 -p squad_ablation_1m -t -1 -g 2 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa

python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/dataset/squad1.1 -p squad_ablation_1m -t -1 -g 1 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa


python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/dataset/squad1.1 -p squad_mask0.05 -t -1 -g 2 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa

python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/dataset/squad1.1 -p squad_mask0.05 -t -1 -g 1 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa


python sweep/sweep_ft_spanqa.py -d /checkpoint/xwhan/uqa -p uqa_bce_squad_valid -t -1 -g 8 -n 4 --tensorboard-logdir /checkpoint/xwhan/spanqa

## evaluation for WebQ
* use kdn model 
```
python scripts/evaluate_reader.py /private/home/xwhan/dataset/WebQ --use-kdn --model-path /checkpoint/xwhan/2019-09-02/WebQ_kdn_m0.05.span_qa.adam.lr5e-06.kdn_v2_mask0.05.seed3.bsz8.ldrop0.1.ngpu1/checkpoint_best.pt -a span_qa


python scripts/evaluate_reader.py /private/home/xwhan/dataset/WebQ --use-kdn --model-path /checkpoint/xwhan/2019-09-02/WebQ_kdn_m0.05.span_qa.adam.lr5e-06.kdn_v2_mask0.05.seed3.bsz8.ldrop0.1.ngpu1/checkpoint_last.pt -a span_qa

```
* use bert model
```
python scripts/evaluate_reader.py /private/home/xwhan/dataset/WebQ --model-path /checkpoint/xwhan/2019-09-02/WebQ_kdn_m0.05.span_qa.adam.lr5e-06.kdn_v2_mask0.05.seed3.bsz8.ldrop0.1.ngpu1/checkpoint_best.pt --arch span_qa 
```

python scripts/evaluate_reader.py /private/home/xwhan/dataset/WebQ --use-kdn --model-path /checkpoint/xwhan/2019-09-09/WebQ_ablation_1m.span_qa.adam.lr5e-06.bert_mlm_ablation.seed3.bsz8.ldrop0.2.ngpu1/checkpoint_best.pt -a span_qa

f1 score 0.3709972196438732
em score 0.30118110236220474

python scripts/evaluate_reader.py /private/home/xwhan/dataset/WebQ --use-kdn --model-path /checkpoint/xwhan/2019-09-09/WebQ_ablation_1m.span_qa.adam.lr5e-06.bert_mlm_ablation.seed3.bsz8.ldrop0.1.ngpu1/checkpoint_last.pt -a span_qa

python scripts/evaluate_reader.py /private/home/xwhan/dataset/WebQ --model-path /checkpoint/xwhan/2019-08-26/WebQ_kdn_v2.span_qa.adam.lr5e-06.kdn_v2_boundary.seed3.bsz8.ldrop0.2.ngpu1/checkpoint_best.pt --arch span_qa



## Evaluation for SQuAD
* KDN model
```
python scripts/evaluate_reader.py /private/home/xwhan/dataset/squad1.1 --model-path /checkpoint/xwhan/2019-08-22/squad_kdn_v2_boundary.span_qa.adam.lr1e-05.kdn_v2_boundary.seed3.bsz8.ldrop0.2.ngpu2/checkpoint4.pt --arch span_qa --eval-data /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json --answer-path /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json


python scripts/evaluate_reader.py /private/home/xwhan/dataset/squad1.1 --model-path /checkpoint/xwhan/2019-09-09/squad_mask0.05.span_qa.adam.lr5e-06.kdn_v2_mask0.05.seed3.bsz8.ldrop0.1.ngpu1/checkpoint_best.pt --arch span_qa --eval-data /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json --answer-path /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json

python scripts/evaluate_reader.py /private/home/xwhan/dataset/squad1.1 --model-path /checkpoint/xwhan/2019-08-23/squad_kdn_v2_boundary_b8.span_qa.adam.lr5e-06.kdn_v2_boundary.seed3.bsz8.ldrop0.2.ngpu1/checkpoint_best.pt --arch span_qa --eval-data /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json --answer-path /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json


python scripts/evaluate_reader.py /private/home/xwhan/dataset/squad1.1 --model-path /checkpoint/xwhan/2019-09-09/squad_mask0.05.span_qa.adam.lr5e-06.kdn_v2_mask0.05.seed3.bsz8.ldrop0.1.ngpu1/checkpoint_best.pt --arch span_qa --eval-data /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json --answer-path /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json



python scripts/evaluate_reader.py /private/home/xwhan/dataset/squad1.1 --model-path /checkpoint/xwhan/2019-09-09/squad_ablation_1m.span_qa.adam.lr5e-06.bert_mlm_ablation.seed3.bsz8.ldrop0.1.ngpu1/checkpoint_best.pt --arch span_qa --eval-data /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json --answer-path /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json



```
* BERT model
python scripts/evaluate_reader.py /private/home/xwhan/dataset/squad1.1 --model-path /checkpoint/xwhan/2019-08-18/squad_bert.span_qa.adam.lr2e-05.bert_best.crs_ent.seed3.bsz8.ldrop0.1.ngpu1/checkpoint_best.pt --arch span_qa --eval-data /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json --answer-path /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json

python scripts/evaluate_reader.py /private/home/xwhan/dataset/squad1.1 --model-path /checkpoint/xwhan/2019-08-18/squad_bert.span_qa.adam.lr5e-06.bert_best.crs_ent.seed3.bsz8.ldrop0.1.ngpu1/checkpoint4.pt --arch span_qa --eval-data /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json --answer-path /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json



# Evaluation for quasart
python scripts/evaluate_reader.py /private/home/xwhan/DrQA/data/datasets/data/datasets/quasart --model-path /checkpoint/xwhan/2019-09-02/quasart_base.span_qa.adam.lr5e-06.bert_best.seed3.bsz32.bsz128.ldrop0.2.ngpu1/checkpoint_best.pt --arch span_qa --eval-data /private/home/xwhan/DrQA/data/datasets/data/datasets/quasart/test_eval.json --answer-path  /private/home/xwhan/DrQA/data/datasets/data/datasets/quasart/test_eval.json

## quasart ablation
python scripts/evaluate_reader.py /private/home/xwhan/DrQA/data/datasets/data/datasets/quasart --model-path /checkpoint/xwhan/2019-09-11/quasart_ablation.span_qa.adam.lr5e-06.bert_mlm_ablation.seed3.bsz32.mlen128.ldrop0.2.ngpu1/checkpoint_best.pt --arch span_qa --eval-data /private/home/xwhan/DrQA/data/datasets/data/datasets/quasart/test_eval.json --answer-path  /private/home/xwhan/DrQA/data/datasets/data/datasets/quasart/test_eval.json

------------------------------------

# Cancel all jobs
squeue -u xwhan | grep 176530 | awk '{print $1}' | xargs -n 1 scancel

------------------------------------
# KDN Pretrainning Experiments
# kdn debug
python train.py --fp16 /checkpoint/xwhan/wiki_data_mlm --task kdn --arch kdn --save-interval 1 --max-update 1000000 --lr 1e-05 --bert-path /checkpoint/jingfeidu/2019-05-28/masked-lm-rand.st512.mt4096.uf1.bert_base.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip1.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu32/checkpoint_best.pt --distributed-world-size 1 --max-sentences 4 --optimizer adam --criterion kdn_loss --ddp-backend no_c10d --save-interval-updates 1000 --use-mlm --last-dropout 0.1 --boundary-loss --save-interval-updates 5
# sweep for kdn v2
python sweep/sweep_ft_kdn.py -d /checkpoint/xwhan/wiki_data_v2 -p kdn_v2_mask0.01 -t -1 -g 8 -n 4 --tensorboard-logdir /checkpoint/xwhan/kdn


python sweep/sweep_ft_kdn.py -d /checkpoint/xwhan/wiki_data_v2 -p kdn_no_mlm -t -1 -g 8 -n 4 --tensorboard-logdir /checkpoint/xwhan/kdn

python sweep/sweep_ft_kdn.py -d /checkpoint/xwhan/wiki_data_v2 -p kdn_no_ent_mlm_max1m -t -1 -g 8 -n 4 --tensorboard-logdir /checkpoint/xwhan/kdn

python sweep/sweep_ft_kdn.py -d /checkpoint/xwhan/wiki_data_v3 -p kdn_v3_boundary_continue -t -1 -g 8 -n 4 --tensorboard-logdir /checkpoint/xwhan/kdn

# sweep for kdn v3
python sweep/sweep_ft_kdn.py -d /checkpoint/xwhan/wiki_data_v3 -p kdn_v3_boundary -t -1 -g 8 -n 4 --tensorboard-logdir /checkpoint/xwhan/kdn

# sweep for kdn m2 k10
python sweep/sweep_ft_kdn.py -d /checkpoint/xwhan/wiki_data_m2_k10 -p kdn_m2_k10_boundary -t -1 -g 8 -n 4 --tensorboard-logdir /checkpoint/xwhan/kdn

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
python scripts/evaluate_re.py --arch re /private/home/xwhan/dataset/tacred --model-path /checkpoint/xwhan/2019-08-15/re_256c_base_uncased_marker.re.adam.lr2e-05.bert.crs_ent.seed3.bsz8.maxlen256.drop0.1.ngpu2/checkpoint_best.pt
```

## sweep relation extraction
python sweep/sweep_ft_re.py -d /private/home/xwhan/dataset/tacred -p re_kdn_v2_bound_marker -t -1 -g 2 -n 1 

# tensorboard logs
ssh -J prn-fairjmp02 -L 8889:localhost:8889 100.97.67.36


## Typing 

python train.py --fp16  /private/home/xwhan/dataset/FIGER --task typing --arch typing --criterion typing_loss --bert-path /checkpoint/jingfeidu/2019-05-28/masked-lm-rand.st512.mt4096.uf1.bert_base.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip1.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu32/checkpoint_best.pt --distributed-world-size 8 --max-sentences 128 --optimizer adam --use-sep --save-interval-updates 10 --ddp-backend no_c10d --save-interval 1 --max-update 1000000 --lr 1e-5 --max-length 64

python sweep/sweep_ft_typing.py -d /private/home/xwhan/dataset/FIGER -p figer_base_marker -t -1 -g 8 -n 2 --tensorboard-logdir /checkpoint/xwhan/typing

python sweep/sweep_ft_typing.py -d /private/home/xwhan/dataset/FIGER -p figer_kdn_marker -t -1 -g 8 -n 2 --tensorboard-logdir /checkpoint/xwhan/typing

python sweep/sweep_ft_typing.py -d /private/home/xwhan/dataset/FIGER -p figer_mask0.15 -t -1 -g 8 -n 2 --tensorboard-logdir /checkpoint/xwhan/typing

python sweep/sweep_ft_typing.py -d /private/home/xwhan/dataset/FIGER -p figer_ablation -t -1 -g 8 -n 2 --tensorboard-logdir /checkpoint/xwhan/typing

python sweep/sweep_ft_typing.py -d /private/home/xwhan/dataset/ontonotes -p Ontonotes_base_marker -t -1 -g 8 -n 2 --tensorboard-logdir /checkpoint/xwhan/typing

python sweep/sweep_ft_typing.py -d /private/home/xwhan/dataset/ontonotes -p Ontonotes_kdn_marker -t -1 -g 8 -n 2 --tensorboard-logdir /checkpoint/xwhan/typing

python scripts/evaluate_typing.py --arch typing /private/home/xwhan/dataset/FIGER --model-path /checkpoint/xwhan/2019-09-12/figer_kdn_marker.typing.adam.lr2e-05.kdn_v2_mask0.05.seed3.maxsent16.maxlen256.drop0.1.ngpu16/checkpoint_best.pt --use-marker --eval-data /private/home/xwhan/dataset/FIGER/processed-splits/test --thresh 0.6 --use-kdn --boundary-loss


## OpenQA Experiments

python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/DrQA/data/datasets/data/datasets/quasart -p quasart_mask0.05 -t -1 -g 1 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa

python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/DrQA/data/datasets/data/datasets/quasart -p quasart_ablation -t -1 -g 1 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa

python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/DrQA/data/datasets/data/datasets/quasart -p quasart_mask0.15 -t -1 -g 1 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa

python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/DrQA/data/datasets/data/datasets/unftriviaqa -p unftriviaqa_mask0.15 -t -1 -g 4 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa

python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/DrQA/data/datasets/data/datasets/unftriviaqa -p unftriviaqa_mask0.05 -t -1 -g 1 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa

python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/DrQA/data/datasets/data/datasets/unftriviaqa -p unftriviaqa_mask0.15 -t -1 -g 1 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa


python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/DrQA/data/datasets/data/datasets/searchqa -p searchqa_mask0.05 -t -1 -g 1 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa

python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/DrQA/data/datasets/data/datasets/searchqa -p searchqa_ablation -t -1 -g 1 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa

python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/DrQA/data/datasets/data/datasets/unftriviaqa -p unftriviaqa_base -t -1 -g 4 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa

python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/DrQA/data/datasets/data/datasets/unftriviaqa -p unftriviaqa_ablation -t -1 -g 1 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa

python sweep/sweep_ft_ranker.py -d /private/home/xwhan/DrQA/data/datasets/data/datasets/searchqa_ranking -p searchqa_ranking_base -t -1 -g 8 -n 1 --tensorboard-logdir /checkpoint/xwhan/ranking 

python sweep/sweep_ft_ranker.py -d /private/home/xwhan/DrQA/data/datasets/data/datasets/unftriviaqa_ranking -p unftriviaqa_ranking_base -t -1 -g 8 -n 2 --tensorboard-logdir /checkpoint/xwhan/ranking 



python train.py --fp16  /private/home/xwhan/DrQA/data/datasets/data/datasets/searchqa_ranking --task paragaph_ranking --arch finetuning_paragraph_ranker --save-interval 1 --max-update 30000 --lr 2e-05 --bert-path /checkpoint/jingfeidu/2019-05-28/masked-lm-rand.st512.mt4096.uf1.bert_base.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip1.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu32/checkpoint_best.pt --num-label 2 --distributed-world-size 8 --max-sentences 32 --optimizer adam --ddp-backend no_c10d --max-length 128


### Ranking eval
python scripts/evaluate_ranker.py /private/home/xwhan/DrQA/data/datasets/data/datasets/searchqa_ranking --model-path /checkpoint/xwhan/2019-09-02/searchqa_ranking_base.finetuning_paragraph_ranker.adam.lr5e-06.bert.crs_ent.seed3.bsz32.bsz128.ldrop0.1.ngpu8/checkpoint_last.pt -a finetuning_paragraph_ranker --eval-data /private/home/xwhan/DrQA/data/datasets/data/datasets/searchqa/valid_eval.json --save-path /private/home/xwhan/DrQA/data/datasets/data/datasets/searchqa/valid_eval_with_scores.json --tokenized


python scripts/evaluate_ranker.py /private/home/xwhan/DrQA/data/datasets/data/datasets/unftriviaqa_ranking --model-path /checkpoint/xwhan/2019-09-03/unftriviaqa_ranking_base.finetuning_paragraph_ranker.adam.lr2e-05.bert.crs_ent.seed3.bsz32.bsz128.ldrop0.1.ngpu16/checkpoint_last.pt -a finetuning_paragraph_ranker --eval-data /private/home/xwhan/DrQA/data/datasets/data/datasets/unftriviaqa/valid_eval.json --save-path /private/home/xwhan/DrQA/data/datasets/data/datasets/unftriviaqa/valid_eval_with_scores.json --tokenized


### Reader eval
python scripts/evaluate_reader.py /private/home/xwhan/DrQA/data/datasets/data/datasets/searchqa --model-path /checkpoint/xwhan/2019-09-03/searchqa_mask0.05.span_qa.adam.lr5e-06.kdn_v2_mask0.05.seed3.bsz32.bsz128.ldrop0.2.ngpu1/checkpoint_best.pt --arch span_qa --eval-data /private/home/xwhan/DrQA/data/datasets/data/datasets/searchqa/test_eval_with_scores.json --answer-path  /private/home/xwhan/DrQA/data/datasets/data/datasets/searchqa/test_eval_with_scores.json


python scripts/evaluate_reader.py /private/home/xwhan/DrQA/data/datasets/data/datasets/unftriviaqa --model-path /checkpoint/xwhan/2019-09-03/unftriviaqa_mask0.05.span_qa.adam.lr5e-06.kdn_v2_mask0.05.seed3.bsz32.bsz128.ldrop0.1.ngpu4/checkpoint_best.pt --arch span_qa --eval-data /private/home/xwhan/DrQA/data/datasets/data/datasets/unftriviaqa/valid_eval_with_scores.json --answer-path  /private/home/xwhan/DrQA/data/datasets/data/datasets/unftriviaqa/valid_eval_with_scores.json

python scripts/evaluate_reader.py /private/home/xwhan/DrQA/data/datasets/data/datasets/unftriviaqa --model-path /checkpoint/xwhan/2019-09-12/unftriviaqa_mask0.15.span_qa.adam.lr5e-06.kdn_v2_boundary.seed3.bsz32.mlen128.ldrop0.2.ngpu1/checkpoint_best.pt --arch span_qa --eval-data /private/home/xwhan/DrQA/data/datasets/data/datasets/unftriviaqa/valid_eval_with_scores.json --answer-path  /private/home/xwhan/DrQA/data/datasets/data/datasets/unftriviaqa/valid_eval_with_scores.json

python scripts/evaluate_reader.py /private/home/xwhan/DrQA/data/datasets/data/datasets/unftriviaqa --model-path /checkpoint/xwhan/2019-09-12/unftriviaqa_ablation.span_qa.adam.lr1e-05.bert_mlm_ablation.seed3.bsz32.mlen128.ldrop0.1.ngpu1/checkpoint_best.pt --arch span_qa --eval-data /private/home/xwhan/DrQA/data/datasets/data/datasets/unftriviaqa/valid_eval_with_scores.json --answer-path  /private/home/xwhan/DrQA/data/datasets/data/datasets/unftriviaqa/valid_eval_with_scores.json



# TODO
### spanqa analysis
python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/DrQA/data/datasets/data/datasets/xxx -p unftriviaqa_mask0.05 -t -1 -g 1 -n 1 --tensorboard-logdir /checkpoint/xwhan/spanqa

### Typing evaluation check

python sweep/sweep_ft_typing.py -d /private/home/xwhan/dataset/ontonotes -p onto_base_marker -t -1 -g 8 -n 2 --tensorboard-logdir /checkpoint/xwhan/typing

