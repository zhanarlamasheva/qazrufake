"""Controls requested in review round 3 (offline, SVM word+char):
AUC and balanced accuracy for every protocol row; balanced-test resampling; size- and balance-matched
standard training; balanced same-publisher training; length-matched original text. Also AUC of
seed-averaged XLM-R and of the zero-shot LLM. Writes results/res_r3.json.
Usage (repo root): python src/analyze_r3.py
"""
import sys, json, collections
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
sys.path.insert(0, 'src')
from lib import model, R

by = {r['uid']: r for r in R}
rng = np.random.default_rng(2026)
UNAMB_FAKE = {'false', 'fake', 'mostly-false'}
S = {
 'ru': lambda r: r['lang'] == 'ru',
 'ru_pub': lambda r: r['lang'] == 'ru' and r['source_name'] != 'Tengrinews.kz',
 'kk': lambda r: r['lang'] == 'kk',
 'kk_pub': lambda r: r['lang'] == 'kk' and r['source_name'] == 'FactcheckKz_kz',
}
S['kk_core'] = lambda r: S['kk_pub'](r) and (r['verdict'] == 'true' or r['verdict'] in UNAMB_FAKE)
split = lambda s, f: [r for r in R if r['split'] == s and f(r)]
nw = lambda t: len(t.split())

def trunc(r):  # original text cut to the word length of its claim-only version
    return ' '.join(r['text'].split()[:max(1, nw(r['claim']))])
for r in R:
    r['trunc'] = trunc(r)

def groups(us):
    g = collections.defaultdict(list)
    for u in us: g[by[u]['group']].append(u)
    return list(g.values())

def gboot(us, pred, score, n=1000):
    gs = groups(us); f, a = [], []
    for _ in range(n):
        s = [u for k in rng.integers(0, len(gs), len(gs)) for u in gs[k]]
        y = [by[u]['label'] for u in s]
        f.append(f1_score(y, [pred[u] for u in s], average='macro'))
        if 0 < sum(y) < len(y): a.append(roc_auc_score(y, [score[u] for u in s]))
    pc = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])]
    return pc(f), pc(a)

def evaluate(us, pred, score, ci=True):
    y = [by[u]['label'] for u in us]; p = [pred[u] for u in us]
    d = dict(n=len(us), n_real=len(y) - sum(y), n_fake=sum(y),
             f1m=float(f1_score(y, p, average='macro')), auc=float(roc_auc_score(y, [score[u] for u in us])),
             bacc=float(balanced_accuracy_score(y, p)),
             majority_f1m=float(f1_score(y, [int(sum(y) * 2 >= len(y))] * len(y), average='macro')))
    if ci: d['f1_ci'], d['auc_ci'] = gboot(us, pred, score)
    return d

def fit(tr, field):
    m = model('SVM', 'both'); m.fit([r[field] for r in tr], [r['label'] for r in tr]); return m

def predict(m, us, field):
    X = [by[u][field] for u in us]
    return dict(zip(us, map(int, m.predict(X)))), dict(zip(us, map(float, m.decision_function(X))))

def balanced_test(us, pred, score, draws=200):
    """Macro-F1 / AUC on class-balanced resamples of the test set (all minority items + equal number of majority)."""
    y = {u: by[u]['label'] for u in us}
    real = [u for u in us if y[u] == 0]; fake = [u for u in us if y[u] == 1]
    k = min(len(real), len(fake)); f = []
    for _ in range(draws):
        s = list(rng.choice(real, k, replace=False)) + list(rng.choice(fake, k, replace=False))
        f.append(f1_score([y[u] for u in s], [pred[u] for u in s], average='macro'))
    return dict(k_per_class=k, f1m_mean=float(np.mean(f)), f1m_p2_5=float(np.percentile(f, 2.5)), f1m_p97_5=float(np.percentile(f, 97.5)))

OUT = {}
ROWS = [  # name, train subset, test subset, field
 ('ru|standard', 'ru', 'ru', 'text'), ('ru|test_restricted', 'ru', 'ru_pub', 'text'),
 ('ru|same_publisher', 'ru_pub', 'ru_pub', 'text'), ('ru|same_publisher_claim', 'ru_pub', 'ru_pub', 'claim'),
 ('ru|same_publisher_trunc', 'ru_pub', 'ru_pub', 'trunc'),
 ('kk|standard', 'kk', 'kk', 'text'), ('kk|test_restricted', 'kk', 'kk_pub', 'text'),
 ('kk|same_publisher', 'kk_pub', 'kk_pub', 'text'), ('kk|same_publisher_claim', 'kk_pub', 'kk_pub', 'claim'),
 ('kk|same_publisher_trunc', 'kk_pub', 'kk_pub', 'trunc'),
 ('kk|core_claim', 'kk_core', 'kk_core', 'claim'), ('kk|core_trunc', 'kk_core', 'kk_core', 'trunc'),
 ('ru|standard_claim', 'ru', 'ru', 'claim'), ('ru|standard_trunc', 'ru', 'ru', 'trunc'),
]
for name, trs, tes, field in ROWS:
    tr = split('train', S[trs]); te = [r['uid'] for r in split('test', S[tes])]
    pr, sc = predict(fit(tr, field), te, field)
    d = evaluate(te, pr, sc); d['train_real'] = sum(1 for r in tr if r['label'] == 0); d['train_fake'] = sum(1 for r in tr if r['label'] == 1)
    d['mean_words_test'] = float(np.mean([nw(by[u][field]) for u in te]))
    d['balanced_test'] = balanced_test(te, pr, sc)
    OUT[name] = d
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k not in ('balanced_test',)}, round(d['balanced_test']['f1m_mean'], 3))

# Control A: standard training (all Russian sources) subsampled to the size and class counts of the same-publisher training set
def matched(trs_pool, n_real, n_fake, tests, field='text', draws=10):
    pool = split('train', S[trs_pool]); real = [r for r in pool if r['label'] == 0]; fake = [r for r in pool if r['label'] == 1]
    res = collections.defaultdict(list)
    for _ in range(draws):
        tr = [real[i] for i in rng.choice(len(real), n_real, replace=False)] + [fake[i] for i in rng.choice(len(fake), n_fake, replace=False)]
        m = fit(tr, field)
        for tn in tests:
            te = [r['uid'] for r in split('test', S[tn])]; pr, sc = predict(m, te, field)
            e = evaluate(te, pr, sc, ci=False); res[tn + '|f1m'].append(e['f1m']); res[tn + '|auc'].append(e['auc'])
    return {k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), min=float(np.min(v)), max=float(np.max(v))) for k, v in res.items()}

nr = OUT['ru|same_publisher']['train_real']; nf = OUT['ru|same_publisher']['train_fake']
OUT['ctrlA|ru|all_sources_matched_size_balance|text'] = matched('ru', nr, nf, ['ru', 'ru_pub'])
OUT['ctrlA|ru|all_sources_matched_size_balance|claim'] = matched('ru', nr, nf, ['ru', 'ru_pub'], field='claim')
# Control B: same-publisher training made class-balanced by undersampling FAKE
OUT['ctrlB|ru|same_publisher_balanced|text'] = matched('ru_pub', nr, nr, ['ru_pub'])
OUT['ctrlB|ru|same_publisher_balanced|claim'] = matched('ru_pub', nr, nr, ['ru_pub'], field='claim')
nr2 = OUT['kk|same_publisher']['train_real']; nf2 = OUT['kk|same_publisher']['train_fake']
OUT['ctrlA|kk|all_sources_matched_size_balance|text'] = matched('kk', nr2, nf2, ['kk', 'kk_pub'])
for k, v in OUT.items():
    if k.startswith('ctrl'): print(k, {a: round(b['mean'], 3) for a, b in v.items()}, {a: round(b['sd'], 3) for a, b in v.items()})

# AUC of seed-averaged XLM-R and of the zero-shot LLM on the same test subsets
def load(path, key):
    P = collections.defaultdict(lambda: collections.defaultdict(list))
    for l in open(path):
        d = json.loads(l); P[d[key]][d['uid']].append(d['prob'])
    return P
XB = load('predictions/xlmr_base_predictions.jsonl', 'setting'); X3 = load('predictions/xlmr_v3_predictions.jsonl', 'setting')
LL = load('predictions/llm_zeroshot_predictions.jsonl', 'field')
XMAP = [('ru|standard', XB, 'ru|text', 'ru'), ('ru|test_restricted', XB, 'ru|text', 'ru_pub'),
        ('ru|same_publisher', X3, 'rufc|text', 'ru_pub'), ('ru|same_publisher_claim', X3, 'rufc|claim', 'ru_pub'),
        ('kk|standard', XB, 'kk|text', 'kk'), ('kk|test_restricted', XB, 'kk|text', 'kk_pub'),
        ('kk|same_publisher', XB, 'kkfc|text', 'kk_pub'), ('kk|same_publisher_claim', XB, 'kkfc|claim', 'kk_pub'),
        ('kk|core_claim', XB, 'kkunamb|claim', 'kk_core')]
for name, P, setting, tes in XMAP:
    if setting not in P: print('missing', setting); continue
    te = [r['uid'] for r in split('test', S[tes])]
    sc = {u: float(np.mean(P[setting][u])) for u in te}; pr = {u: int(sc[u] > 0.5) for u in te}
    OUT['xlmr_seedavg|' + name] = evaluate(te, pr, sc)
for field in ['text', 'claim']:
    for tes in ['ru', 'ru_pub', 'kk', 'kk_pub', 'kk_core']:
        te = [r['uid'] for r in split('test', S[tes])]
        sc = {u: LL[field][u][0] for u in te}; pr = {u: int(sc[u] > 0.5) for u in te}
        OUT[f'llm|{field}|{tes}'] = evaluate(te, pr, sc)
for k, v in OUT.items():
    if k.startswith(('xlmr', 'llm')): print(k, round(v['f1m'], 3), round(v['auc'], 3), [round(x, 3) for x in v['auc_ci']])
json.dump(OUT, open('results/res_r3.json', 'w'), indent=1, ensure_ascii=False)
