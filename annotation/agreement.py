"""Usage: python annotation/agreement.py [annotation/double_annotation_295.csv]

Cohen's kappa and raw agreement between the linguist and the fact-checker on the
295-item double-annotation sample, per language, for fake-news type and
disinformation technique (results/res_agreement.json, Table 2b).

For technique, the 15 items where the fact-checker entered a fake-news type label in
the technique field are excluded, as described in the paper. The four-group coarse
type agreement in res_agreement.json is not recomputed here.
"""
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score

path = sys.argv[1] if len(sys.argv) > 1 else "annotation/double_annotation_295.csv"
d = pd.read_csv(path)
types = set(d.type_linguist) | set(d.type_factchecker)
for lang in ["ru", "kk", "all"]:
    s = d if lang == "all" else d[d.lang == lang]
    t = s[~s.technique_factchecker.isin(types)]
    for name, frame, a, b in [("type", s, "type_linguist", "type_factchecker"),
                              ("technique", t, "technique_linguist", "technique_factchecker")]:
        x, y = frame[a].astype(str), frame[b].astype(str)
        print(f"{lang:3s} {name:9s} n={len(frame):3d}  agreement={(x == y).mean():.3f}  "
              f"kappa={cohen_kappa_score(x, y):.3f}")
