import json,re,collections,numpy as np,random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score,precision_recall_fscore_support
R=[json.loads(l) for l in open('data/qazrufake.jsonl', encoding='utf-8')]
rng=np.random.default_rng(0)
def boot(y,p,n=1000):
    y=np.array(y);p=np.array(p);s=[]
    for _ in range(n):
        i=rng.integers(0,len(y),len(y)); s.append(f1_score(y[i],p[i],average='macro'))
    return np.percentile(s,[2.5,97.5])
def sel(lang=None,split=None,f=None):
    return [r for r in R if (lang is None or r['lang'] in lang) and (split is None or r['split']==split) and (f is None or f(r))]
def feats(kind):
    if kind=='word': return TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,lowercase=True)
    if kind=='char': return TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=2,sublinear_tf=True,max_features=300000)
    if kind=='both': return make_union(feats('word'),feats('char'))
def model(name,kind):
    if name=='LR': clf=LogisticRegression(C=10,max_iter=3000,class_weight='balanced')
    if name=='SVM': clf=LinearSVC(C=0.5,class_weight='balanced')
    if name=='NB': clf=ComplementNB(alpha=0.3)
    return make_pipeline(feats(kind),clf)
def run(tr,te,name='LR',kind='char',textf=lambda r:r['text']):
    m=model(name,kind); m.fit([textf(r) for r in tr],[r['label'] for r in tr])
    X=[textf(r) for r in te]; y=[r['label'] for r in te]; p=m.predict(X)
    try: s=m.decision_function(X)
    except: s=m.predict_proba(X)[:,1]
    lo,hi=boot(y,p)
    return dict(n=len(te),acc=accuracy_score(y,p),f1m=f1_score(y,p,average='macro'),f1fake=f1_score(y,p),auc=roc_auc_score(y,s),ci=(lo,hi)),m,p
def fmt(d): return f"n={d['n']} acc={d['acc']:.3f} F1m={d['f1m']:.3f} [{d['ci'][0]:.3f},{d['ci'][1]:.3f}] F1fake={d['f1fake']:.3f} AUC={d['auc']:.3f}"
