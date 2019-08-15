# Paragraph Ranker Experiemnts
## debug para ranker
python train.py --fp16  /private/home/xwhan/dataset/webq_ranking --task paragaph_ranking --arch finetuning_paragraph_ranker --save-nterval 1 --max-update 30000 --lr 2e-05 --bert-path /checkpoint/jingfeidu/2019-05-28/masked-lm-rand.st512.mt4096.uf1.bert_base.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip1.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu32/checkpoint_best.pt --num-label 2 --distributed-world-size 1 --max-sentences 8 --optimizer adam
## sweep for paragraph ranking
python sweep/sweep_ft_ranker.py -d /private/home/xwhan/dataset/webq_ranking -p ranker_balanced_ldrop -t -1 -g 1 -n 1 --partition dev

------------------------------------

# Span QA Experiments
## debug SQuAD
python train.py --fp16  /private/home/xwhan/dataset/squad1.1 --task span_qa --arch span_qa --save-interval 1 --max-update 30000 --lr 1e-05 --bert-path /checkpoint/xwhan/2019-08-13/kdn_pred_on_ends.adam.bert.crs_ent.seed3.bsz4.0.01.lr1e-05.ngpu32/checkpoint_best.pt --distributed-world-size 1 --max-sentences 8 --optimizer adam --criterion span_qa --use-kdn --start-end
## debug 
python train.py --fp16  /private/home/xwhan/dataset/webq_qa --task span_qa --arch span_qa --save-interval 1 --max-update 30000 --lr 1e-05 --bert-path /checkpoint/xwhan/2019-08-13/kdn_pred_on_ends.adam.bert.crs_ent.seed3.bsz4.0.01.lr1e-05.ngpu32/checkpoint_best.pt --distributed-world-size 1 --max-sentences 8 --optimizer adam --criterion span_qa --use-kdn --start-end

## sweep for WebQuesions

```python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/dataset/webq_qa -p webq_kdn -t -1 -g 1 -n 1```

## sweep for SQuAd1.1 

python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/dataset/squad1.1 -p squad_kdn -t -1 -g 2 -n 1
python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/dataset/squad1.1 -p squad_bert -t -1 -g 2 -n 1

## evaluation for WebQ
* use kdn model 
```
python scripts/evaluate_reader.py /private/home/xwhan/dataset/webq_qa --use-kdn --model-path /checkpoint/xwhan/2019-08-13/webq_kdn.span_qa.adam.lr1e-05.kdn_best.crs_ent.seed3.bsz8.ngpu1/checkpoint_last.pt --start-end -a span_qa
```
* use bert model
```
python scripts/evaluate_reader.py /private/home/xwhan/dataset/webq_qa --model-path /checkpoint/xwhan/2019-08-13/squad_kdn.span_qa.adam.lr1e-05.kdn_best.crs_ent.seed3.bsz8.ngpu1/checkpoint_best.pt --arch span_qa 
```

## Evaluation for SQuAD
* KDN model
```
python scripts/evaluate_reader.py /private/home/xwhan/dataset/squad1.1 --model-path /checkpoint/xwhan/2019-08-13/squad_kdn_.span_qa.adam.lr1e-05.kdn_best.crs_ent.seed3.bsz8.ngpu1/checkpoint_best.pt --arch span_qa --use-kdn --start-end --eval-data /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json --answer-path /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json
```
* BERT model
python scripts/evaluate_reader.py /private/home/xwhan/dataset/squad1.1 --model-path /checkpoint/xwhan/2019-08-13/squad_bert.span_qa.adam.lr1e-05.bert_best.crs_ent.seed3.bsz8.ngpu1/checkpoint_best.pt --arch span_qa --eval-data /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json --answer-path /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json

------------------------------------

# Cancel all jobs
squeue -u xwhan | grep 167 | awk '{print $1}' | xargs -n 1 scancel

------------------------------------
# KDN Pretrainning Experiments
# kdn debug
python train.py --fp16 /checkpoint/xwhan/wiki_data --task kdn --arch kdn --save-interval 1 --max-update 1000000 --lr 1e-05 --bert-path /checkpoint/jingfeidu/2019-05-28/masked-lm-rand.st512.mt4096.uf1.bert_base.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip1.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu32/checkpoint_best.pt --distributed-world-size 1 --max-sentences 8 --optimizer adam --criterion kdn_loss --ddp-backend no_c10d --save-interval-updates 1000 --use-mlm --last-dropout 0.1
# sweep for kdn
python sweep/sweep_ft_kdn.py -d /checkpoint/xwhan/wiki_data -p kdn_pred_on_ends -t -1 -g 8 -n 4

# data processing flow
* replace the entities, in process_wiki, `python process_wikipedia.py`
* split into 512 chunks, in fairseq-py, `python scripts/preprocess_kdn.py`
* binarize the data with sweep

------------------------------------
# Relation Extraction Experiments

## debug relation extraction
/checkpoint/ves/2019-05-31/mlm-big-bookwiki.st512.mt4096.uf1.bert_large.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip4.0.adam_eps6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu64/checkpoint_best.pt

/checkpoint/jingfeidu/2019-05-28/masked-lm-rand.st512.mt4096.uf1.bert_base.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip1.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu32/checkpoint_best.pt

```python train.py --fp16  /private/home/xwhan/dataset/tacred --task re --arch re --save-interval 1 --max-update 30000 --lr 2e-05 --bert-path /checkpoint/jingfeidu/2019-05-28/masked-lm-rand.st512.mt4096.uf1.bert_base.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip1.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu32/checkpoint_best.pt --distributed-world-size 1 --max-sentences 8 --optimizer adam --criterion cross_entropy --use-marker --curriculum 1 --ddp-backend no_c10d```

```use
python train.py --fp16  /private/home/xwhan/dataset/tacred --task re --arch re --save-interval 1 --max-update 30000 --lr 2e-05 --bert-path /checkpoint/ves/2019-05-31/mlm-big-bookwiki.st512.mt4096.uf1.bert_large.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip4.0.adam_eps6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu64/checkpoint_best.pt --distributed-world-size 1 --max-sentences 8 --optimizer adam --criterion cross_entropy --use-marker --curriculum 1 --model-dim 1024 --ddp-backend no_c10d --use-hf --use-cased
```

## evaluate relation extraction
Use BERT model
```
python scripts/evaluate_re.py --arch re /private/home/xwhan/dataset/tacred 
```

## sweep relation extraction
python sweep/sweep_ft_re.py -d /private/home/xwhan/dataset/tacred -p re_base_uncased_marker -t -1 -g 1 -n 1 

# tensorboard logs
ssh -J prn-fairjmp02 -L 8889:localhost:8889 100.97.67.36


