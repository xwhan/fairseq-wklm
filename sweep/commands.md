# Paragraph Ranker Experiemnts
## debug para ranker
python train.py --fp16  /private/home/xwhan/dataset/webq_ranking --task paragaph_ranking --arch finetuning_paragraph_ranker --save-nterval 1 --max-update 30000 --lr 2e-05 --bert-path /checkpoint/jingfeidu/2019-05-28/masked-lm-rand.st512.mt4096.uf1.bert_base.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip1.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu32/checkpoint_best.pt --num-label 2 --distributed-world-size 1 --max-sentences 8 --optimizer adam
## sweep for paragraph ranking
python sweep/sweep_ft_ranker.py -d /private/home/xwhan/dataset/webq_ranking -p ranker_balanced_ldrop -t -1 -g 1 -n 1 --partition dev

------------------------------------

# Span QA Experiments
## debug span_qa
python train.py --fp16  /private/home/xwhan/dataset/webq_qa --task span_qa --arch span_qa --save-interval 1 --max-update 30000 --lr 2e-05 --bert-path /checkpoint/xwhan/2019-08-07/kdn_start_end.adam.bert.crs_ent.seed3.bsz8.0.01.lr1e-05.beta998.warmup10000.ngpu16/checkpoint_best.pt --distributed-world-size 1 --max-sentences 8 --optimizer adam --criterion span_qa --use-kdn
## sweep for span qa
python sweep/sweep_ft_spanqa.py -d /private/home/xwhan/dataset/webq_qa -p reader_ft -t -1 -g 2 -n 1
## evaluation for span_qa
* use kdn model 
```
python scripts/evaluate_reader.py /private/home/xwhan/dataset/webq_qa --use-kdn --model-path /checkpoint/xwhan/2019-08-08/reader_ft.span_qa.mxup187500.adam.lr1e-05.kdn_last.crs_ent.seed3.bsz8.ngpu2/checkpoint_last.pt --save --sabe
```
* use bert model
```
python scripts/evaluate_reader.py /private/home/xwhan/dataset/webq_qa --model-path /checkpoint/xwhan/2019-07-10/reader_fix_binarize.span_qa.mxup61875.adam.lr1e-05.bert.crs_ent.seed3.bsz8.ngpu1/checkpoint_last.pt --arch span_qa --save --save_name pred_ft_bert_only_ans_score.json
```

------------------------------------

# cancel all jobs
squeue -u xwhan | grep 163 | awk '{print $1}' | xargs -n 1 scancel

------------------------------------
# KDN Pretrainning Experiments
# kdn debug
python train.py --fp16 /checkpoint/xwhan/wiki_data --task kdn --arch kdn --save-interval 1 --max-update 1000000 --lr 1e-05 --bert-path /checkpoint/jingfeidu/2019-05-28/masked-lm-rand.st512.mt4096.uf1.bert_base.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip1.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu32/checkpoint_best.pt --distributed-world-size 1 --max-sentences 8 --optimizer adam --criterion kdn_loss --ddp-backend no_c10d --save-interval-updates 1000 --use-mlm
# sweep for kdn
python sweep/sweep_ft_kdn.py -d /checkpoint/xwhan/wiki_data -p kdn_pred_on_start -t -1 -g 8 -n 2

# data processing flow
* replace the entities, in process_wiki, `python process_wikipedia.py`
* split into 512 chunks, in fairseq-py, `python scripts/preprocess_kdn.py`
* binarize the data with sweep

------------------------------------
# Relation Extraction Experiments

## debug relation extraction
```
python train.py --fp16  /private/home/xwhan/dataset/tacred --task re --arch re --save-interval 1 --max-update 30000 --lr 2e-05 --bert-path /checkpoint/jingfeidu/2019-05-28/masked-lm-rand.st512.mt4096.uf1.bert_base.dr0.1.atdr0.1.actdr0.1.wd0.01.adam.beta998.clip1.0.clip6e-06.lr0.0001.warm10000.fp16.mu3000000.seed1.ngpu32/checkpoint_best.pt --distributed-world-size 1 --max-sentences 8 --optimizer adam --criterion cross_entropy --use-marker --curriculum 1
```

## evaluate relation extraction
Use BERT model
```
python scripts/evaluate_re.py --arch re /private/home/xwhan/dataset/tacred 
```

## sweep relation extraction
python sweep/sweep_ft_re.py -d /private/home/xwhan/dataset/tacred -p re_marker_cls -t -1 -g 1 -n 1 

# tensorboard logs
ssh -J prn-fairjmp02 -L 8889:localhost:8889 100.97.67.36


