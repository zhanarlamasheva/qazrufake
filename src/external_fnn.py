"""Apply the QazRuFake evaluation protocol to FakeNewsNet (PolitiFact and GossipCop; Shu et al., 2020).
Data: the four CSVs from github.com/KaiDMML/FakeNewsNet/tree/master/dataset (id, news_url, title, tweet_ids).
Rows: standard (title text), metadata-only probe (publisher domain), test restricted to publishers with both
labels, same-publisher train+test, and a publisher-disjoint split. Claim-only text does not apply (titles only).
Usage: python src/external_fnn.py <dir with the four CSVs>   -> results/res_external_fnn.json
"""
import sys, re, json, collections
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
sys.path.insert(0, 'src')
from lib import model

rng = np.random.default_rng(2026)
D = sys.argv[1]

def domain(u):
    u = str(u).strip().lower()
    m = re.search(r'archive\.(?:org|is|today|ph)/(?:web/)?[^/]*/(.+)$', u)  # archived copy: use the original URL
    if m and '.' in re.sub(r'^https?:/+', '', m.group(1)).split('/')[0]:
        u = m.group(1)
    u = re.sub(r'^https?:/+', '', u); u = re.sub(r'^www\d?\.', '', u)
    h = re.sub(r'^www\d?\.', '', u.split('/')[0].split('?')[0].split(':')[0])
    parts = h.split('.')
    return '.'.join(parts[-3:]) if len(parts) > 2 and parts[-2] in ('co', 'com', 'org', 'net', 'gov', 'ac') else '.'.join(parts[-2:])

def norm(t): return re.sub(r'[^a-z0-9]+', ' ', str(t).lower()).strip()

def gboot(y, p, s, g, n=1000):
    idx = collections.defaultdict(list)
    for i, k in enumerate(g): idx[k].append(i)
    G = list(idx.values()); f, a = [], []
    for _ in range(n):
        ii = [i for k in rng.integers(0, len(G), len(G)) for i in G[k]]
        yy = y[ii]; f.append(f1_score(yy, p[ii], average='macro'))
        if 0 < yy.sum() < len(yy): a.append(roc_auc_score(yy, s[ii]))
    pc = lambda v: [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]
    return pc(f), pc(a)

def ev(te, p, s):
    y = te.label.values; p = np.asarray(p); s = np.asarray(s)
    fc, ac = gboot(y, p, s, te.group.values)
    maj = int(y.mean() >= 0.5)
    return dict(n=len(te), n_real=int((y == 0).sum()), n_fake=int(y.sum()), f1m=round(float(f1_score(y, p, average='macro')), 3), f1_ci=fc,
                auc=round(float(roc_auc_score(y, s)), 3), auc_ci=ac, majority_f1m=round(float(f1_score(y, [maj] * len(y), average='macro')), 3))

def text_row(tr, te):
    m = model('SVM', 'both'); m.fit(tr.title.tolist(), tr.label.values)
    return ev(te, m.predict(te.title.tolist()), m.decision_function(te.title.tolist()))

def meta_row(tr, te):
    enc = OneHotEncoder(handle_unknown='ignore'); Xtr = enc.fit_transform(tr[['domain']]); Xte = enc.transform(te[['domain']])
    m = LogisticRegression(C=10, max_iter=3000, class_weight='balanced').fit(Xtr, tr.label.values)
    d = ev(te, m.predict(Xte), m.decision_function(Xte)); d['test_domain_seen_in_train'] = round(float(te.domain.isin(set(tr.domain)).mean()), 3)
    return d

OUT = {}
for ds in ['politifact', 'gossipcop']:
    df = pd.concat([pd.read_csv(f'{D}/{ds}_fake.csv').assign(label=1), pd.read_csv(f'{D}/{ds}_real.csv').assign(label=0)])
    n0 = len(df)
    df = df.dropna(subset=['news_url', 'title']); df['domain'] = df.news_url.map(domain); df = df[df.domain.str.contains(r'\.')]
    df['tnorm'] = df.title.map(norm); df = df[df.tnorm.str.len() > 0]
    # groups: identical normalised titles (and identical URLs) form one group, so near-copies never straddle the split
    gid = {}; df['group'] = [gid.setdefault(t, len(gid)) for t in df.tnorm]
    mixed = df.groupby('group').label.nunique().gt(1).sum()
    groups = np.array(df.group.unique()); rng.shuffle(groups); cut = int(0.8 * len(groups)); trg = set(groups[:cut])
    df['split'] = np.where(df.group.isin(trg), 'train', 'test')
    both = set(df.groupby('domain').label.nunique().loc[lambda s: s > 1].index)
    tr, te = df[df.split == 'train'], df[df.split == 'test']
    trB, teB = tr[tr.domain.isin(both)], te[te.domain.isin(both)]
    R = dict(n_raw=n0, n_used=len(df), n_domains=int(df.domain.nunique()), n_domains_both_labels=len(both),
             share_items_from_both_label_domains=round(float(df.domain.isin(both).mean()), 3), mixed_label_title_groups=int(mixed),
             top_domains_real=df[df.label == 0].domain.value_counts().head(5).round(3).to_dict(),
             top_domains_fake=df[df.label == 1].domain.value_counts().head(5).to_dict())
    R['standard'] = text_row(tr, te)
    R['metadata_probe_domain'] = meta_row(tr, te)
    R['test_restricted'] = text_row(tr, teB)
    R['same_publisher'] = text_row(trB, teB)
    R['same_publisher_metadata_probe'] = meta_row(trB, teB)
    # publisher-disjoint split: 80% of domains for training, 20% for test
    doms = np.array(df.domain.unique(), dtype=object); rng.shuffle(doms); dtr = set(doms[:int(0.8 * len(doms))])
    trD = df[df.domain.isin(dtr) & (df.split == 'train')]; teD = df[~df.domain.isin(dtr) & (df.split == 'test')]
    if teD.label.nunique() == 2: R['publisher_disjoint'] = text_row(trD, teD)
    OUT[ds] = R
    print(ds, json.dumps({k: v for k, v in R.items() if not k.startswith('top')}, ensure_ascii=False)[:1500])
    print(' top real', R['top_domains_real']); print(' top fake', R['top_domains_fake'])
json.dump(OUT, open('results/res_external_fnn.json', 'w'), indent=1)
