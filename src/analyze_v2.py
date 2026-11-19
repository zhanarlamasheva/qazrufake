"""Offline analysis of Colab v2 outputs.
Usage (from repo root): python src/analyze_v2.py predictions/xlmr_base_predictions.jsonl predictions/labse_embeddings.npz
Produces res_v2.json: XLM-R mean±SD per setting/subset, paired group-bootstrap
XLM-R − SVM differences on identical items, LaBSE bilingual-duplicate flags and
cross-lingual results with flagged items removed."""
import sys, json, collections
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

R = {json.loads(l)['uid']: json.loads(l) for l in open('data/qazrufake.jsonl', encoding='utf-8')}
X = [json.loads(l) for l in open(sys.argv[1])]
SVM = json.load(open('predictions/svm_wordchar_predictions.json'))
rng = np.random.default_rng(0)
OUT = {}

is_fc = lambda r: r['fc_kk']
is_un = lambda r: r['unamb']
SUBSETS = {'all': lambda r: True, 'fc': is_fc, 'unamb': is_un,
           'ru_fcout': lambda r: r['lang'] == 'ru' and r['source_name'] != 'Tengrinews.kz'}

# --- XLM-R per setting, seed-level F1 and seed-averaged probabilities
xp = collections.defaultdict(dict)   # (setting, uid) -> {seed: prob}
for d in X:
    xp[d['setting']].setdefault(d['uid'], {})[d['seed']] = d['prob']


def f1(uids, prob):
    y = [R[u]['label'] for u in uids]
    return f1_score(y, [int(prob[u] > 0.5) for u in uids], average='macro')


def eval_setting(setting, test_lang, subset):
    P = xp[setting]
    uids = [u for u in P if R[u]['lang'] == test_lang and SUBSETS[subset](R[u])]
    seeds = sorted({s for u in uids for s in P[u]})
    per_seed = [f1(uids, {u: P[u][s] for u in uids}) for s in seeds]
    mean_prob = {u: float(np.mean(list(P[u].values()))) for u in uids}
    return dict(n=len(uids), f1_mean=float(np.mean(per_seed)), f1_sd=float(np.std(per_seed, ddof=1)) if len(seeds) > 1 else None,
                auc=float(roc_auc_score([R[u]['label'] for u in uids], [mean_prob[u] for u in uids])), seeds=len(seeds)), mean_prob, uids


def paired(uids, pa, pb, n=1000):
    """paired group bootstrap of macro-F1(a) - macro-F1(b)"""
    groups = collections.defaultdict(list)
    for u in uids:
        groups[R[u]['group']].append(u)
    keys = list(groups)
    y = {u: R[u]['label'] for u in uids}
    diffs = []
    for _ in range(n):
        s = [u for k in rng.integers(0, len(keys), len(keys)) for u in groups[keys[k]]]
        yy = [y[u] for u in s]
        diffs.append(f1_score(yy, [pa[u] for u in s], average='macro') - f1_score(yy, [pb[u] for u in s], average='macro'))
    return dict(mean=float(np.mean(diffs)), ci=[float(x) for x in np.percentile(diffs, [2.5, 97.5])],
                p_le0=float(np.mean(np.array(diffs) <= 0)))


svm = collections.defaultdict(dict)
for d in SVM:
    svm[(d['field'], d['train'], d['test'])][d['uid']] = d['pred']

# settings comparable with SVM predictions saved in exp_rev.py (train on ru / kk / ru,kk)
for setting, field, train in [('ru|text', 'text', 'ru'), ('ru|claim', 'claim', 'ru'), ('kk|text', 'text', 'kk'),
                              ('kk|claim', 'claim', 'kk'), ('ru,kk|text', 'text', 'ru,kk')]:
    for test in ['ru', 'kk']:
        for subset in ['all', 'fc', 'unamb', 'ru_fcout']:
            if (subset in ('fc', 'unamb') and test != 'kk') or (subset == 'ru_fcout' and test != 'ru'):
                continue
            res, mp, uids = eval_setting(setting, test, subset)
            sp = svm[(field, train, test)]
            res['paired_xlmr_minus_svm'] = paired(uids, {u: int(mp[u] > 0.5) for u in uids}, sp)
            OUT[f'{setting}->{test}|{subset}'] = res
            print(f"{setting:12s}->{test} {subset:8s} n={res['n']:4d} XLM-R {res['f1_mean']:.3f}±{(res['f1_sd'] or 0):.3f}  "
                  f"Δ vs SVM {res['paired_xlmr_minus_svm']['mean']:+.3f} {res['paired_xlmr_minus_svm']['ci']}")

# restricted-training settings (no SVM preds saved for these: compare F1 only)
for setting in ['kkfc|text', 'kkfc|claim', 'kkunamb|text', 'kkunamb|claim']:
    subset = 'fc' if setting.startswith('kkfc') else 'unamb'
    res, mp, uids = eval_setting(setting, 'kk', subset)
    OUT[f'{setting}->kk|{subset}'] = res
    print(f"{setting:14s}->kk {subset:6s} n={res['n']} XLM-R {res['f1_mean']:.3f}±{(res['f1_sd'] or 0):.3f}")

# --- LaBSE bilingual duplicates
if len(sys.argv) > 2:
    Z = np.load(sys.argv[2])
    E = Z['emb'].astype('float32'); U = list(Z['uid'])
    ru = [i for i, u in enumerate(U) if u.startswith('ru')]; kk = [i for i, u in enumerate(U) if u.startswith('kk')]
    S = E[ru] @ E[kk].T
    OUT['labse_sim_quantiles'] = [float(x) for x in np.percentile(S.max(1), [50, 90, 99])]
    for th in [0.80, 0.75, 0.70]:
        pairs = np.argwhere(S >= th)
        links = collections.defaultdict(set)
        for a, b in pairs:
            links[U[ru[a]]].add(U[kk[b]]); links[U[kk[b]]].add(U[ru[a]])
        OUT[f'labse_links_{th}'] = dict(pairs=int(len(pairs)), ru=sum(1 for k in links if k.startswith('ru')),
                                        kk=sum(1 for k in links if k.startswith('kk')))
        for src, setting, test in [('ru', 'ru|text', 'kk'), ('kk', 'kk|text', 'ru')]:
            seen = {u for u, r in R.items() if r['lang'] == src and r['split'] in ('train', 'dev')}
            res, mp, uids = eval_setting(setting, test, 'all')
            clean = [u for u in uids if not (links[u] & seen)]
            P = xp[setting]
            per_seed = [f1(clean, {u: P[u][s] for u in clean}) for s in sorted({s for u in clean for s in P[u]})]
            OUT[f'xling_labse{th}_{setting}->{test}'] = dict(n=len(clean), removed=len(uids) - len(clean),
                                                             f1_mean=float(np.mean(per_seed)), f1_sd=float(np.std(per_seed, ddof=1)))
            print('LaBSE', th, setting, '->', test, OUT[f'xling_labse{th}_{setting}->{test}'])
json.dump(OUT, open('results/res_v2.json', 'w'), indent=1)
