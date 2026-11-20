"""Offline analysis of the Colab v3 outputs (XLM-R on Russian fact-checking outlets; zero-shot LLM).
Usage (from repo root):
    python src/analyze_v3.py predictions/xlmr_v3_predictions.jsonl predictions/llm_zeroshot_predictions.jsonl
Writes results/res_v3.json: XLM-R per seed and seed-averaged, zero-shot LLM macro-F1 with group-bootstrap CIs,
and paired group-bootstrap differences against SVMs retrained on the same training sets.
"""
import sys, json, collections
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
sys.path.insert(0, 'src')
from lib import model, R
from neutralize import claim_only

for r in R:
    r['claim'] = claim_only(r['text'])
by = {r['uid']: r for r in R}
rng = np.random.default_rng(0)
UNAMB_FAKE = {'false', 'fake', 'mostly-false'}
is_fc_kk = lambda r: r['lang'] == 'kk' and r['source_name'] == 'FactcheckKz_kz'
is_unamb = lambda r: is_fc_kk(r) and (r['verdict'] == 'true' or r['verdict'] in UNAMB_FAKE)
SUB = {'ru|all': lambda r: r['lang'] == 'ru',
       'ru|fcout': lambda r: r['lang'] == 'ru' and r['source_name'] != 'Tengrinews.kz',
       'kk|all': lambda r: r['lang'] == 'kk', 'kk|fc': is_fc_kk, 'kk|unamb': is_unamb}


def groups_of(uids):
    g = collections.defaultdict(list)
    for u in uids:
        g[by[u]['group']].append(u)
    return list(g.values())


def gboot(uids, pred, n=1000):
    gs = groups_of(uids); y = {u: by[u]['label'] for u in uids}; out = []
    for _ in range(n):
        s = [u for k in rng.integers(0, len(gs), len(gs)) for u in gs[k]]
        out.append(f1_score([y[u] for u in s], [pred[u] for u in s], average='macro'))
    return [float(x) for x in np.percentile(out, [2.5, 97.5])]


def paired(uids, pa, pb, n=1000):
    gs = groups_of(uids); y = {u: by[u]['label'] for u in uids}; d = []
    for _ in range(n):
        s = [u for k in rng.integers(0, len(gs), len(gs)) for u in gs[k]]
        yy = [y[u] for u in s]
        d.append(f1_score(yy, [pa[u] for u in s], average='macro') - f1_score(yy, [pb[u] for u in s], average='macro'))
    return dict(mean=float(np.mean(d)), ci=[float(x) for x in np.percentile(d, [2.5, 97.5])])


def f1(uids, pred):
    return float(f1_score([by[u]['label'] for u in uids], [pred[u] for u in uids], average='macro'))


def svm_preds(train_filter, field, test_uids):
    tr = [r for r in R if r['split'] == 'train' and train_filter(r)]
    m = model('SVM', 'both'); m.fit([r[field] for r in tr], [r['label'] for r in tr])
    return dict(zip(test_uids, (int(p) for p in m.predict([by[u][field] for u in test_uids]))))


OUT = {}
test = [r['uid'] for r in R if r['split'] == 'test']

# --- XLM-R trained and tested on the Russian fact-checking outlets
X = [json.loads(l) for l in open(sys.argv[1])]
xp = collections.defaultdict(lambda: collections.defaultdict(dict))
for d in X:
    xp[d['setting']][d['uid']][d['seed']] = d['prob']
fc_ru = [u for u in test if SUB['ru|fcout'](by[u])]
for setting, field in [('rufc|text', 'text'), ('rufc|claim', 'claim')]:
    P = xp[setting]
    seeds = sorted({s for u in fc_ru for s in P[u]})
    per_seed = []
    for s in seeds:
        pr = {u: int(P[u][s] > 0.5) for u in fc_ru}
        per_seed.append(dict(seed=s, f1m=f1(fc_ru, pr), pred_fake=float(np.mean(list(pr.values()))),
                             auc=float(roc_auc_score([by[u]['label'] for u in fc_ru], [P[u][s] for u in fc_ru]))))
    avg = {u: float(np.mean(list(P[u].values()))) for u in fc_ru}
    pa = {u: int(avg[u] > 0.5) for u in fc_ru}
    sv = svm_preds(SUB['ru|fcout'], field, fc_ru)
    f = [d['f1m'] for d in per_seed]
    OUT[f'xlmr|{setting}'] = dict(n=len(fc_ru), per_seed=per_seed, f1_mean=float(np.mean(f)), f1_sd=float(np.std(f, ddof=1)),
                                  f1_seedavg=f1(fc_ru, pa), svm_f1=f1(fc_ru, sv), paired_xlmr_minus_svm=paired(fc_ru, pa, sv))

# --- zero-shot LLM
L = [json.loads(l) for l in open(sys.argv[2])]
lp = collections.defaultdict(dict)
for d in L:
    lp[d['field']][d['uid']] = d['prob']
core = [u for u in test if is_unamb(by[u])]
svm_core = svm_preds(is_unamb, 'claim', core)
for field in ['text', 'claim']:
    for name, filt in SUB.items():
        us = [u for u in test if filt(by[u])]
        pr = {u: int(lp[field][u] > 0.5) for u in us}
        OUT[f'llm|{field}|{name}'] = dict(n=len(us), f1m=f1(us, pr), ci=gboot(us, pr),
                                          auc=float(roc_auc_score([by[u]['label'] for u in us], [lp[field][u] for u in us])),
                                          pred_fake=float(np.mean(list(pr.values()))),
                                          true_fake=float(np.mean([by[u]['label'] for u in us])))
llm_core = {u: int(lp['claim'][u] > 0.5) for u in core}
OUT['core|svm_minus_llm'] = paired(core, svm_core, llm_core)
OUT['core|svm_f1'] = f1(core, svm_core)
json.dump(OUT, open('results/res_v3.json', 'w'), indent=1)
for k, v in OUT.items():
    print(k, json.dumps(v)[:400])
