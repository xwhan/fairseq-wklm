#for f in /checkpoint/jingfeidu/2019-09-17/triviaqa_nomask.span_qa.adam.lr*/checkpoint_best.pt;
#for f in /checkpoint/jingfeidu/2019-09-17/squad.span_qa.adam.lr*/checkpoint_best.pt;
#for f in /checkpoint/jingfeidu/2019-09-17/unftriviaqa_mask0.05.span_qa.adam.lr*/checkpoint_best.pt
for f in /checkpoint/jingfeidu/2019-09-17/squad.span_qa.adam.lr*/checkpoint_best.pt
do
  #python scripts/evaluate_reader.py /private/home/xwhan/DrQA/data/datasets/data/datasets/quasart --model-path $f --arch span_qa --eval-data /private/home/xwhan/DrQA/data/datasets/data/datasets/quasart/test_eval_with_scores.json --answer-path  /private/home/xwhan/DrQA/data/datasets/data/datasets/quasart/test_eval_with_scores.json

  #python scripts/evaluate_reader.py /private/home/xwhan/DrQA/data/datasets/data/datasets/unftriviaqa --model-path $f --arch span_qa --eval-data /private/home/xwhan/DrQA/data/datasets/data/datasets/unftriviaqa/valid_eval_with_scores.json --answer-path  /private/home/xwhan/DrQA/data/datasets/data/datasets/unftriviaqa/valid_eval_with_scores.json
  python scripts/evaluate_reader.py /private/home/xwhan/dataset/squad1.1 --model-path $f --arch span_qa --eval-data /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json  --answer-path  /private/home/xwhan/dataset/squad1.1/splits/valid_eval.json 
done
