"""Fine-tune multilingual transformers on the QazRuFake corpus (same splits as the paper).

Run on Colab / any GPU machine:
    pip install -q transformers datasets accelerate scikit-learn
    python run_transformers.py --model xlm-roberta-base --train ru --test ru
    python run_transformers.py --model xlm-roberta-base --train ru --test kk     # zero-shot cross-lingual
    python run_transformers.py --model xlm-roberta-base --train ru,kk --test kk   # joint

Suggested models: xlm-roberta-base, bert-base-multilingual-cased, xlm-roberta-large,
DeepPavlov/rubert-base-cased (RU only), kz-transformers/kaz-roberta-conversational (KK only).
Results are appended to results_transformers.jsonl. Use 3 seeds (--seed 1/2/3) and report mean ± sd.
"""
import argparse, json, numpy as np, torch
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='xlm-roberta-base')
ap.add_argument('--train', default='ru')
ap.add_argument('--test', default='ru')
ap.add_argument('--corpus', default='data/qazrufake.jsonl')
ap.add_argument('--epochs', type=float, default=4)
ap.add_argument('--lr', type=float, default=2e-5)
ap.add_argument('--bs', type=int, default=16)
ap.add_argument('--maxlen', type=int, default=256)
ap.add_argument('--seed', type=int, default=1)
a = ap.parse_args()
set_seed(a.seed)

R = [json.loads(l) for l in open(a.corpus, encoding='utf-8')]
tr_l, te_l = a.train.split(','), a.test.split(',')
pick = lambda langs, split: [r for r in R if r['lang'] in langs and r['split'] == split]
tr, dev, te = pick(tr_l, 'train'), pick(tr_l, 'dev'), pick(te_l, 'test')

tok = AutoTokenizer.from_pretrained(a.model)
def ds(rs):
    d = Dataset.from_dict({'text': [r['text'] for r in rs], 'label': [r['label'] for r in rs]})
    return d.map(lambda b: tok(b['text'], truncation=True, max_length=a.maxlen), batched=True)
dtr, ddev, dte = ds(tr), ds(dev), ds(te)

model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2)
def metrics(p):
    y = p.label_ids; pr = p.predictions.argmax(-1)
    return {'f1m': f1_score(y, pr, average='macro'), 'acc': accuracy_score(y, pr)}
args = TrainingArguments(output_dir='out', learning_rate=a.lr, per_device_train_batch_size=a.bs,
    per_device_eval_batch_size=64, num_train_epochs=a.epochs, weight_decay=0.01, warmup_ratio=0.1,
    eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True,
    metric_for_best_model='f1m', save_total_limit=1, fp16=torch.cuda.is_available(),
    report_to='none', seed=a.seed)
t = Trainer(model=model, args=args, train_dataset=dtr, eval_dataset=ddev,
            data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
t.train()
p = t.predict(dte)
y = p.label_ids; logits = p.predictions; pr = logits.argmax(-1)
prob = torch.softmax(torch.tensor(logits), -1)[:, 1].numpy()
res = dict(model=a.model, train=a.train, test=a.test, seed=a.seed, n=len(te),
           acc=accuracy_score(y, pr), f1m=f1_score(y, pr, average='macro'),
           f1fake=f1_score(y, pr), auc=roc_auc_score(y, prob))
print(res)
with open('results_transformers.jsonl', 'a') as f:
    f.write(json.dumps(res) + '\n')
