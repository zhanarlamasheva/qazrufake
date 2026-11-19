"""Revision experiments (SVM word+char) answering reviewer comments 1-5."""
import json, re, collections, datetime, math
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from lib import model, sel, R
from neutralize import claim_only, STOP

rng = np.random.default_rng(0)
OUT = {}
for r in R:
    r['claim'] = claim_only(r['text'])
UNAMB_FAKE = {'false', 'fake', 'mostly-false'}
is_fc_kk = lambda r: r['lang'] == 'kk' and r['source_name'] == 'FactcheckKz_kz'
is_unamb = lambda r: is_fc_kk(r) and (r['verdict'] == 'true' or r['verdict'] in UNAMB_FAKE)


def comp(rs):
    c = collections.Counter(r['label'] for r in rs)
    return f"{len(rs)} ({c[0]} R / {c[1]} F)"


def fit(tr, field='text'):
    m = model('SVM', 'both')
    m.fit([r[field] for r in tr], [r['label'] for r in tr])
    return m


def group_boot_ci(te, p, n=1000):
    """Bootstrap macro-F1 over article groups."""
    groups = collections.defaultdict(list)
    for i, r in enumerate(te):
        groups[r['group']].append(i)
    keys = list(groups)
    y = np.array([r['label'] for r in te]); p = np.array(p)
    s = []
    for _ in range(n):
        idx = np.concatenate([groups[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        s.append(f1_score(y[idx], p[idx], average='macro'))
    return [float(x) for x in np.percentile(s, [2.5, 97.5])]


def ev(m, te, field='text'):
    X = [r[field] for r in te]; y = [r['label'] for r in te]
    p = m.predict(X); s = m.decision_function(X)
    d = dict(n=len(te), comp=comp(te), acc=accuracy_score(y, p), f1m=f1_score(y, p, average='macro'),
             f1fake=f1_score(y, p, zero_division=0), auc=roc_auc_score(y, s) if len(set(y)) > 1 else None,
             ci=group_boot_ci(te, p))
    return d, p, s


def show(name, d):
    OUT[name] = d
    auc = f"{d['auc']:.3f}" if d.get('auc') is not None else '-'
    print(f"{name:55s} {d['comp']:24s} acc {d['acc']:.3f} F1m {d['f1m']:.3f} [{d['ci'][0]:.3f},{d['ci'][1]:.3f}] F1f {d['f1fake']:.3f} AUC {auc}")


# ---------- 1. label basis ----------
basis = collections.Counter((r['lang'], r['label'], r['source_name'], r['verdict'] or '-') for r in R)
OUT['label_basis'] = [list(k) + [v] for k, v in sorted(basis.items())]

# ---------- 2. standard + unambiguous + per verdict (KK) ----------
kk_tr, kk_te = sel(['kk'], 'train'), sel(['kk'], 'test')
m_kk = fit(kk_tr)
show('KK std: train all / test all', ev(m_kk, kk_te)[0])
show('KK std: train all / test FC only', ev(m_kk, [r for r in kk_te if is_fc_kk(r)])[0])
show('KK std: train all / test FC unambiguous', ev(m_kk, [r for r in kk_te if is_unamb(r)])[0])
m_u = fit([r for r in kk_tr if is_unamb(r)])
show('KK unamb: train FC-unamb / test FC-unamb', ev(m_u, [r for r in kk_te if is_unamb(r)])[0])
m_fc = fit([r for r in kk_tr if is_fc_kk(r)])
show('KK FC: train FC / test FC', ev(m_fc, [r for r in kk_te if is_fc_kk(r)])[0])
# per-verdict accuracy for the two FC models
pv = {}
for nm, m in [('train_all', m_kk), ('train_FC', m_fc), ('train_unamb', m_u)]:
    te = [r for r in kk_te if is_fc_kk(r)]
    p = m.predict([r['text'] for r in te])
    tab = collections.defaultdict(lambda: [0, 0])
    for r, q in zip(te, p):
        tab[r['verdict']][0] += int(q == r['label']); tab[r['verdict']][1] += 1
    pv[nm] = {k: (c / n, n) for k, (c, n) in tab.items()}
OUT['per_verdict'] = pv
print('per-verdict', json.dumps({k: {v: f"{a:.2f}/{n}" for v, (a, n) in d.items()} for k, d in pv.items()}))

# ---------- 3. claim-only (neutralised) ----------
for lang in ['ru', 'kk']:
    rs = sel([lang])
    for lab in [0, 1]:
        sub = [r for r in rs if r['label'] == lab]
        OUT[f'cue_{lang}_{lab}'] = dict(orig=float(np.mean([bool(STOP.search(r['text'])) for r in sub])),
                                        claim=float(np.mean([bool(STOP.search(r['claim'])) for r in sub])),
                                        words_orig=float(np.mean([len(r['text'].split()) for r in sub])),
                                        words_claim=float(np.mean([len(r['claim'].split()) for r in sub])))
print('cues', {k: {a: round(b, 2) for a, b in v.items()} for k, v in OUT.items() if k.startswith('cue_')})
settings = {
    'RU all': (lambda r: r['lang'] == 'ru', lambda r: r['lang'] == 'ru'),
    'RU fact-check outlets only': (lambda r: r['lang'] == 'ru' and r['source_name'] != 'Tengrinews.kz',) * 2,
    'KK all': (lambda r: r['lang'] == 'kk',) * 2,
    'KK FC only': (is_fc_kk,) * 2,
    'KK FC unambiguous': (is_unamb,) * 2,
}
for nm, (ftr, fte) in settings.items():
    tr = [r for r in R if r['split'] == 'train' and ftr(r)]
    te = [r for r in R if r['split'] == 'test' and fte(r)]
    for field in ['text', 'claim']:
        show(f'{nm} [{field}]', ev(fit(tr, field), te, field)[0])

# ---------- 4. restricted-source variants with explicit composition ----------
mixed_ru = lambda r: r['lang'] == 'ru' and r['source_name'] != 'Tengrinews.kz'
ru_tr, ru_te = sel(['ru'], 'train'), sel(['ru'], 'test')
m_ru = fit(ru_tr)
show('RU std: train all / test all', ev(m_ru, ru_te)[0])
show('RU: train all / test mixed-label sources', ev(m_ru, [r for r in ru_te if mixed_ru(r)])[0])
show('RU: train mixed / test mixed', ev(fit([r for r in ru_tr if mixed_ru(r)]), [r for r in ru_te if mixed_ru(r)])[0])
OUT['comp_train'] = {'RU all': comp(ru_tr), 'RU mixed': comp([r for r in ru_tr if mixed_ru(r)]),
                     'KK all': comp(kk_tr), 'KK FC': comp([r for r in kk_tr if is_fc_kk(r)]),
                     'KK unamb': comp([r for r in kk_tr if is_unamb(r)])}
print('train comps', OUT['comp_train'])

# ---------- 5. temporal split (KK), group-consistent ----------
gdates = collections.defaultdict(list)
for r in sel(['kk']):
    if r['date']:
        gdates[r['group']].append(r['date'][:4])
cross = {g for g, ys in gdates.items() if min(ys) <= '2023' < max(ys)}
kk_dated = [r for r in sel(['kk']) if r['date'] and r['group'] not in cross]
ttr = [r for r in kk_dated if r['date'][:4] <= '2023']; tte = [r for r in kk_dated if r['date'][:4] >= '2024']
OUT['temporal_protocol'] = dict(undated_excluded=sum(1 for r in sel(['kk']) if not r['date']),
                                crossing_groups=len(cross),
                                crossing_items=sum(1 for r in sel(['kk']) if r['group'] in cross),
                                train=comp(ttr), test=comp(tte))
print('temporal', OUT['temporal_protocol'])
show('KK temporal (train<=2023, test>=2024)', ev(fit(ttr), tte)[0])
show('KK temporal FC-only', ev(fit([r for r in ttr if is_fc_kk(r)]), [r for r in tte if is_fc_kk(r)])[0])

# ---------- 6. within-language near-duplicate sensitivity ----------
from sklearn.feature_extraction.text import TfidfVectorizer
for lang in ['ru', 'kk']:
    tr, te = sel([lang], 'train'), sel([lang], 'test')
    v = TfidfVectorizer(analyzer='char_wb', ngram_range=(4, 5), min_df=2, sublinear_tf=True).fit([r['text'] for r in tr + te])
    S = (v.transform([r['text'] for r in te]) @ v.transform([r['text'] for r in tr]).T).max(axis=1).toarray().ravel()
    m = m_ru if lang == 'ru' else m_kk
    OUT[f'maxsim_{lang}'] = dict(p50=float(np.median(S)), p90=float(np.percentile(S, 90)), max=float(S.max()))
    print(lang, 'max train similarity of test items', OUT[f'maxsim_{lang}'])
    for t in [0.5, 0.4, 0.3]:
        sub = [r for r, s in zip(te, S) if s < t]
        show(f'{lang.upper()} test items with max train sim < {t}', ev(m, sub)[0])

# ---------- 7. cross-lingual with bilingual-duplicate filtering ----------
P = json.load(open('results/xling_pairs.json'))
Rd = {r['uid']: r for r in R}
def dd(a, b):
    try: return abs((datetime.date.fromisoformat(a) - datetime.date.fromisoformat(b)).days)
    except Exception: return 9999
links = collections.defaultdict(set)
for s, sh, a, b in P:
    d = dd(Rd[a]['date'], Rd[b]['date'])
    if (s >= 0.15 and d <= 3) or (s >= 0.5 and d <= 14):
        links[a].add(b); links[b].add(a)
OUT['xling_links'] = dict(pairs=sum(len(v) for k, v in links.items() if k.startswith('ru')),
                          ru_items=sum(1 for k in links if k.startswith('ru')), kk_items=sum(1 for k in links if k.startswith('kk')))
print('xling links', OUT['xling_links'])
for a, b in [('ru', 'kk'), ('kk', 'ru')]:
    tr = sel([a], 'train'); te = sel([b], 'test')
    trset = {r['uid'] for r in sel([a], 'train') + sel([a], 'dev')}
    m = model('SVM', 'char'); m.fit([r['text'] for r in tr], [r['label'] for r in tr])
    clean = [r for r in te if not (links[r['uid']] & trset)]
    OUT[f'xling_{a}_{b}_removed'] = len(te) - len(clean)
    for nm, sub in [('all', te), ('linked removed', clean)]:
        X = [r['text'] for r in sub]; y = [r['label'] for r in sub]; p = m.predict(X)
        d = dict(n=len(sub), comp=comp(sub), acc=accuracy_score(y, p), f1m=f1_score(y, p, average='macro'),
                 f1fake=f1_score(y, p), auc=roc_auc_score(y, m.decision_function(X)), ci=group_boot_ci(sub, p))
        show(f'xling SVM-char {a}->{b} {nm}', d)

# ---------- 8. save SVM per-item predictions for paired comparison with XLM-R ----------
preds = []
for train in ['ru', 'kk', 'ru,kk']:
    langs = train.split(',')
    tr = sel(langs, 'train')
    for field in ['text', 'claim']:
        m = fit(tr, field)
        for test in ['ru', 'kk']:
            te = sel([test], 'test')
            s = m.decision_function([r[field] for r in te])
            for r, x in zip(te, s):
                preds.append(dict(model='svm-wordchar', field=field, train=train, test=test, seed=0, uid=r['uid'], score=float(x), pred=int(x > 0)))
json.dump(preds, open('predictions/svm_wordchar_predictions.json', 'w'))
json.dump({k: {'uid': k, 'links': sorted(v)} for k, v in links.items()}, open('results/xling_links.json', 'w'))
json.dump(OUT, open('results/res_rev.json', 'w'), indent=1, default=float, ensure_ascii=False)
