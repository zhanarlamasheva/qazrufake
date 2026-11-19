"""Paired SVM vs XLM-R on restricted training sets, LaBSE-based duplicate checks (within and across languages).
Usage (from repo root): python src/analyze_v2_extra.py"""
import json, collections, numpy as np, sys
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
sys.argv = ['x', 'predictions/xlmr_base_predictions.jsonl']
exec(open('src/analyze_v2.py').read().split('svm = collections.defaultdict')[0])
from lib import model
Rl = list(R.values()); out = {}
for setting, filt, sub, field in [('kkfc|text', lambda r: r['fc_kk'], 'fc', 'text'), ('kkfc|claim', lambda r: r['fc_kk'], 'fc', 'claim'),
                                  ('kkunamb|text', lambda r: r['unamb'], 'unamb', 'text'), ('kkunamb|claim', lambda r: r['unamb'], 'unamb', 'claim')]:
    tr = [r for r in Rl if r['split'] == 'train' and filt(r)]
    m = model('SVM', 'both'); m.fit([r[field] for r in tr], [r['label'] for r in tr])
    res, mp, uids = eval_setting(setting, 'kk', sub)
    sp = dict(zip(uids, m.predict([R[u][field] for u in uids])))
    out[setting] = dict(xlmr=res, svm=f1_score([R[u]['label'] for u in uids], [sp[u] for u in uids], average='macro'),
                        paired=paired(uids, {u: int(mp[u] > 0.5) for u in uids}, sp))
    print(setting, out[setting]['svm'], res['f1_mean'], out[setting]['paired'])
Z = np.load('predictions/labse_embeddings.npz'); E = Z['emb'].astype('float32'); U = list(Z['uid']); idx = {u: i for i, u in enumerate(U)}
ruI = [i for i, u in enumerate(U) if u.startswith('ru')]; kkI = [i for i, u in enumerate(U) if u.startswith('kk')]
S = E[ruI] @ E[kkI].T
svm = collections.defaultdict(dict)
for d in json.load(open('predictions/svm_wordchar_predictions.json')):
    if d['field'] == 'text': svm[(d['train'], d['test'])][d['uid']] = d['pred']
for lang, setting in [('ru', 'ru|text'), ('kk', 'kk|text')]:
    te = [u for u in U if R[u]['lang'] == lang and R[u]['split'] == 'test']; tr = [u for u in U if R[u]['lang'] == lang and R[u]['split'] == 'train']
    mx = (E[[idx[u] for u in te]] @ E[[idx[u] for u in tr]].T).max(1)
    clean = [u for u, s in zip(te, mx) if s < 0.80]; y = [R[u]['label'] for u in clean]
    out[f'within_{lang}'] = dict(removed=len(te) - len(clean), xlmr=float(np.mean([f1_score(y, [int(xp[setting][u][s] > 0.5) for u in clean], average='macro') for s in [1, 2, 3]])),
                                 svm=f1_score(y, [svm[(lang, lang)][u] for u in clean], average='macro'))
    print('within', lang, out[f'within_{lang}'])
for th in [0.80, 0.75]:
    L = collections.defaultdict(set)
    for a, b in np.argwhere(S >= th): L[U[ruI[a]]].add(U[kkI[b]]); L[U[kkI[b]]].add(U[ruI[a]])
    for src, setting, test in [('ru', 'ru|text', 'kk'), ('kk', 'kk|text', 'ru')]:
        seen = {u for u, r in R.items() if r['lang'] == src and r['split'] in ('train', 'dev')}
        clean = [u for u in xp[setting] if R[u]['lang'] == test and not (L[u] & seen)]; y = [R[u]['label'] for u in clean]
        f = [f1_score(y, [int(xp[setting][u][s] > 0.5) for u in clean], average='macro') for s in [1, 2, 3]]
        out[f'xling_labse{th}_{src}->{test}'] = dict(n=len(clean), f1_mean=float(np.mean(f)), f1_sd=float(np.std(f, ddof=1)))
        print(th, src, test, out[f'xling_labse{th}_{src}->{test}'])
json.dump(out, open('results/res_v2_extra.json', 'w'), indent=1, default=float)
