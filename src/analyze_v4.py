"""Round-3 GPU reruns: XLM-R base with a class-weighted loss (5 seeds; 10 for the Russian same-publisher claim-only row and
QazRuFake-Core) and Qwen2.5-14B-Instruct (4-bit) prompted zero-shot in the language of the text, threshold chosen on dev.
Inputs: predictions/xlmr_v4_classweighted_predictions.jsonl, predictions/llm_v4_qwen14b_native_predictions.jsonl
(produced by notebooks/qazrufake_v4.ipynb). Output: results/res_v4.json.   Usage: python src/analyze_v4.py
"""
import sys, json, collections
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
sys.path.insert(0, 'src')
from lib import model, R

by = {r['uid']: r for r in R}
rng = np.random.default_rng(2026)
S = {'ru': lambda r: r['lang'] == 'ru', 'ru_pub': lambda r: r['lang'] == 'ru' and r['source_name'] != 'Tengrinews.kz',
     'kk': lambda r: r['lang'] == 'kk', 'kk_pub': lambda r: r['lang'] == 'kk' and r['fc_kk'],
     'kk_core': lambda r: r['lang'] == 'kk' and r['unamb']}
test = lambda sub: [r['uid'] for r in R if r['split'] == 'test' and S[sub](r)]
y_ = lambda us: [by[u]['label'] for u in us]
f1 = lambda us, p: float(f1_score(y_(us), [p[u] for u in us], average='macro'))

def groups(us):
    g = collections.defaultdict(list)
    for u in us: g[by[u]['group']].append(u)
    return list(g.values())
def boot(us, p, s, n=1000):
    G = groups(us); f, a = [], []
    for _ in range(n):
        ss = [u for k in rng.integers(0, len(G), len(G)) for u in G[k]]; yy = y_(ss)
        f.append(f1_score(yy, [p[u] for u in ss], average='macro')); a.append(roc_auc_score(yy, [s[u] for u in ss]))
    pc = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])]
    return pc(f), pc(a)
def paired(us, pa, pb, n=1000):
    G = groups(us); d = []
    for _ in range(n):
        ss = [u for k in rng.integers(0, len(G), len(G)) for u in G[k]]; yy = y_(ss)
        d.append(f1_score(yy, [pa[u] for u in ss], average='macro') - f1_score(yy, [pb[u] for u in ss], average='macro'))
    return dict(mean=float(np.mean(d)), ci=[float(x) for x in np.percentile(d, [2.5, 97.5])])

TRAIN = {'ru|text': ('ru', 'text'), 'kk|text': ('kk', 'text'), 'rufc|text': ('ru_pub', 'text'), 'rufc|claim': ('ru_pub', 'claim'),
         'kkfc|text': ('kk_pub', 'text'), 'kkfc|claim': ('kk_pub', 'claim'), 'kkunamb|claim': ('kk_core', 'claim')}
svm_cache = {}
def svm(setting, sub):
    if (setting, sub) not in svm_cache:
        trs, field = TRAIN[setting]
        tr = [r for r in R if r['split'] == 'train' and S[trs](r)]
        m = model('SVM', 'both'); m.fit([r[field] for r in tr], [r['label'] for r in tr])
        us = test(sub); X = [by[u][field] for u in us]
        svm_cache[(setting, sub)] = (dict(zip(us, map(int, m.predict(X)))), dict(zip(us, map(float, m.decision_function(X)))))
    return svm_cache[(setting, sub)]

X = collections.defaultdict(lambda: collections.defaultdict(dict))
for l in open('predictions/xlmr_v4_classweighted_predictions.jsonl'):
    d = json.loads(l); X[d['setting']][d['seed']][d['uid']] = d['prob']
OUT = {}
ROWS = [('ru|text', 'ru'), ('ru|text', 'ru_pub'), ('rufc|text', 'ru_pub'), ('rufc|claim', 'ru_pub'),
        ('kk|text', 'kk'), ('kk|text', 'kk_pub'), ('kkfc|text', 'kk_pub'), ('kkfc|claim', 'kk_pub'), ('kkunamb|claim', 'kk_core')]
for setting, sub in ROWS:
    us = test(sub); seeds = sorted(X[setting])
    per = [f1(us, {u: int(X[setting][s][u] > 0.5) for u in us}) for s in seeds]
    share_fake = [float(np.mean([X[setting][s][u] > 0.5 for u in us])) for s in seeds]
    avg = {u: float(np.mean([X[setting][s][u] for s in seeds])) for u in us}; pavg = {u: int(avg[u] > 0.5) for u in us}
    fci, aci = boot(us, pavg, avg)
    sp, ss = svm(setting, sub)
    OUT[f'xlmr_cw|{setting}|{sub}'] = dict(n_seeds=len(seeds), f1_mean=float(np.mean(per)), f1_sd=float(np.std(per, ddof=1)),
        f1_per_seed=[round(v, 3) for v in per], min_share_pred_fake=min(share_fake), max_share_pred_fake=max(share_fake),
        seedavg_f1=f1(us, pavg), seedavg_f1_ci=fci, seedavg_auc=float(roc_auc_score(y_(us), [avg[u] for u in us])), seedavg_auc_ci=aci,
        svm_f1=f1(us, sp), paired_xlmr_minus_svm=paired(us, pavg, sp))
    o = OUT[f'xlmr_cw|{setting}|{sub}']
    print(f"XLM-R cw {setting:14s} on {sub:8s} {o['f1_mean']:.3f}±{o['f1_sd']:.3f} n={o['n_seeds']} seedavg {o['seedavg_f1']:.3f} AUC {o['seedavg_auc']:.3f} "
          f"SVM {o['svm_f1']:.3f} paired {o['paired_xlmr_minus_svm']['mean']:+.3f} {[round(x,3) for x in o['paired_xlmr_minus_svm']['ci']]} fake-share {o['min_share_pred_fake']:.2f}-{o['max_share_pred_fake']:.2f}")

L = collections.defaultdict(dict)
for l in open('predictions/llm_v4_qwen14b_native_predictions.jsonl'):
    d = json.loads(l); L[(d['field'], d['split'])][d['uid']] = d['prob']
LROWS = [('text', 'ru', 'ru|text'), ('text', 'ru_pub', 'rufc|text'), ('claim', 'ru_pub', 'rufc|claim'), ('text', 'kk', 'kk|text'),
         ('text', 'kk_pub', 'kkfc|text'), ('claim', 'kk_pub', 'kkfc|claim'), ('claim', 'kk_core', 'kkunamb|claim')]
for field, sub, svm_setting in LROWS:
    dv = [r['uid'] for r in R if r['split'] == 'dev' and S[sub](r)]; us = test(sub)
    pd_ = L[(field, 'dev')]; pt = L[(field, 'test')]
    grid = np.unique(np.quantile([pd_[u] for u in dv], np.linspace(0.01, 0.99, 99)))
    thr = float(max(grid, key=lambda t: f1_score(y_(dv), [int(pd_[u] > t) for u in dv], average='macro')))
    p05 = {u: int(pt[u] > 0.5) for u in us}; pdv = {u: int(pt[u] > thr) for u in us}; sc = {u: pt[u] for u in us}
    fci, aci = boot(us, pdv, sc)
    sp, _ = svm(svm_setting, sub)
    OUT[f'llm14b|{field}|{sub}'] = dict(n=len(us), thr_dev=thr, f1_thr05=f1(us, p05), f1_thrdev=f1(us, pdv), f1_thrdev_ci=fci,
        auc=float(roc_auc_score(y_(us), [pt[u] for u in us])), auc_ci=aci, share_pred_fake_thrdev=float(np.mean(list(pdv.values()))),
        share_fake_true=float(np.mean(y_(us))), svm_f1=f1(us, sp), paired_svm_minus_llm=paired(us, sp, pdv))
    o = OUT[f'llm14b|{field}|{sub}']
    print(f"LLM14B {field:5s} {sub:8s} thr {thr:.4f} F1@.5 {o['f1_thr05']:.3f} F1@dev {o['f1_thrdev']:.3f} {[round(x,3) for x in fci]} AUC {o['auc']:.3f} "
          f"{[round(x,3) for x in aci]} predFAKE {o['share_pred_fake_thrdev']:.2f}/{o['share_fake_true']:.2f} SVM-LLM {o['paired_svm_minus_llm']['mean']:+.3f} "
          f"{[round(x,3) for x in o['paired_svm_minus_llm']['ci']]}")
json.dump(OUT, open('results/res_v4.json', 'w'), indent=1)
