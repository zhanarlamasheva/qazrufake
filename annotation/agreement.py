"""Usage: python agreement.py double_annotation_sample.xlsx first_annotation_key.csv
Computes Cohen's kappa and raw agreement for fake_type and technique, per language."""
import sys, pandas as pd
from sklearn.metrics import cohen_kappa_score
b = pd.read_excel(sys.argv[1], sheet_name='annotate')
a = pd.read_csv(sys.argv[2])
d = a.merge(b, on='uid')
d = d[(d.fake_type.astype(str).str.strip() != '') & d.fake_type.notna()]
for lang in ['ru', 'kk', 'all']:
    s = d if lang == 'all' else d[d.lang == lang]
    for col in ['fake_type', 'technique']:
        x, y = s[col + '_A'].astype(str), s[col].astype(str)
        print(f"{lang:3s} {col:10s} n={len(s):3d}  agreement={(x == y).mean():.3f}  kappa={cohen_kappa_score(x, y):.3f}")
