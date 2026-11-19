from lib import *
out={}
# --- top features RU / KK
for lang in ['ru','kk']:
    tr=sel([lang],'train')
    m=model('LR','word'); m.fit([r['text'] for r in tr],[r['label'] for r in tr])
    v=m[0].get_feature_names_out(); c=m[1].coef_[0]; o=np.argsort(c)
    out[f'top_{lang}']=dict(fake=list(v[o[::-1][:25]]),real=list(v[o[:25]]))
    print(lang,'FAKE:',', '.join(v[o[::-1][:25]]));print(lang,'REAL:',', '.join(v[o[:25]]))
# --- metadata-only baselines
from sklearn.preprocessing import OneHotEncoder
for lang in ['ru','kk']:
    tr=sel([lang],'train');te=sel([lang],'test')
    for feat in ['source_name','year','source_type']:
        g=lambda r: r['source_name'] if feat=='source_name' else (r['date'][:4] or 'NA') if feat=='year' else (r['source_type'] or 'NA')
        tab=collections.defaultdict(collections.Counter)
        for r in tr: tab[g(r)][r['label']]+=1
        glob=collections.Counter(r['label'] for r in tr).most_common(1)[0][0]
        p=[tab[g(r)].most_common(1)[0][0] if g(r) in tab else glob for r in te]; y=[r['label'] for r in te]
        out[f'meta_{lang}_{feat}']=dict(acc=accuracy_score(y,p),f1m=f1_score(y,p,average='macro'))
        print('meta',lang,feat,out[f'meta_{lang}_{feat}'])
    # length
    from sklearn.tree import DecisionTreeClassifier
    X=lambda rs:[[len(r['text']),len(r['text'].split()),r['text'].count('«')+r['text'].count('"'),sum(ch.isdigit() for ch in r['text'])] for r in rs]
    dt=DecisionTreeClassifier(max_depth=4,random_state=0).fit(X(tr),[r['label'] for r in tr]); p=dt.predict(X(te));y=[r['label'] for r in te]
    out[f'meta_{lang}_surface']=dict(acc=accuracy_score(y,p),f1m=f1_score(y,p,average='macro')); print('meta',lang,'surface',out[f'meta_{lang}_surface'])
# --- within-source evaluation (RU): only sources that have both labels
for lang in ['ru','kk']:
    tr=sel([lang],'train')
    both={s for s,c in collections.Counter((r['source_name'],r['label']) for r in R if r['lang']==lang).items()}
    srcs={s for (s,l) in both if (s,1-l) in both}
    print(lang,'mixed-label sources',srcs)
    d,m,_=run(tr,sel([lang],'test',lambda r:r['source_name'] in srcs),'SVM','both'); out[f'within_{lang}']=d; print('within-source test',lang,fmt(d))
    if lang=='ru':
        d,_,_=run(sel([lang],'train',lambda r:r['source_name'] in srcs),sel([lang],'test',lambda r:r['source_name'] in srcs),'SVM','both'); out['within_ru_trainmixed']=d; print('within-source train+test ru',fmt(d))
        print('train label dist mixed', collections.Counter(r['label'] for r in sel([lang],'train',lambda r:r['source_name'] in srcs)))
# --- cue masking
CUES_RU=r"(сообщени\w* распространял\w*[^.]*\.|в (социальных сетях|соцсетях|мессенджер\w*|сети|публикациях)\w*|распространя\w+|утвержда\w+|якобы|фейк\w*|опроверг\w*|ложн\w*|манипуляц\w*|недостоверн\w*|не соответству\w*|whatsapp|telegram|tiktok|instagram|facebook)"
CUES_KK=r"(әлеуметтік желі\w*|желі\w*|тарап\w*|тарат\w*|жалған|фейк\w*|манипуляци\w*|шындыққа сәйкес\w*|whatsapp|telegram|tiktok|instagram|facebook|делінген|деп жаз\w*)"
def mask(r):
    t=r['text']; t=re.sub(CUES_RU if r['lang']=='ru' else CUES_KK,' ',t,flags=re.I); return t
for lang in ['ru','kk']:
    rate=np.mean([mask(r)!=r['text'] for r in sel([lang]) if r['label']==1]),np.mean([mask(r)!=r['text'] for r in sel([lang]) if r['label']==0])
    print(lang,'cue present fake/real',rate); out[f'cue_rate_{lang}']=rate
    d,_,_=run(sel([lang],'train'),sel([lang],'test'),'SVM','both',textf=mask); out[f'masked_{lang}']=d; print('masked',lang,fmt(d))
    # train normal, test masked
    m=model('SVM','both'); tr=sel([lang],'train'); m.fit([r['text'] for r in tr],[r['label'] for r in tr]); te=sel([lang],'test')
    p=m.predict([mask(r) for r in te]); y=[r['label'] for r in te]
    out[f'masked_testonly_{lang}']=dict(f1m=f1_score(y,p,average='macro')); print('train-orig test-masked',lang,f1_score(y,p,average='macro'))
json.dump(out,open('results/res_E3.json','w'),indent=1,default=float,ensure_ascii=False)
