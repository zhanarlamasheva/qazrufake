"""Tables 4-5: in-language baselines, cross-lingual and joint training (SVM/LR/NB)."""
from lib import *
out={}
def fmt(d): return f"n={d['n']} acc={d['acc']:.3f} F1m={d['f1m']:.3f} [{d['ci'][0]:.3f},{d['ci'][1]:.3f}] F1fake={d['f1fake']:.3f} AUC={d['auc']:.3f}"
# E1 in-language
for lang in ['ru','kk']:
    tr=sel([lang],'train');te=sel([lang],'test')
    y=[r['label'] for r in te]; maj=[collections.Counter(r['label'] for r in tr).most_common(1)[0][0]]*len(te)
    out[f'E1_{lang}_majority']=dict(n=len(te),acc=accuracy_score(y,maj),f1m=f1_score(y,maj,average='macro'))
    print(lang,'majority',out[f'E1_{lang}_majority'])
    for name,kind in [('NB','word'),('LR','word'),('LR','char'),('SVM','word'),('SVM','char'),('SVM','both')]:
        d,_,_=run(tr,te,name,kind); out[f'E1_{lang}_{name}_{kind}']=d; print(lang,name,kind,fmt(d))
# E2 cross-lingual / joint
for a,b in [('ru','kk'),('kk','ru')]:
    d,_,_=run(sel([a],'train'),sel([b],'test'),'SVM','char'); out[f'E2_{a}->{b}']=d; print('xling',a,b,fmt(d))
for lang in ['ru','kk']:
    d,_,_=run(sel(['ru','kk'],'train'),sel([lang],'test'),'SVM','char'); out[f'E2_joint->{lang}']=d; print('joint',lang,fmt(d))
json.dump(out,open('results/res_E1E2.json','w'),indent=1,default=float)
