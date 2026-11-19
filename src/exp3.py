from lib import *
out={}
# per-group breakdown with SVM both
for lang in ['ru','kk']:
    tr=sel([lang],'train');te=sel([lang],'test')
    d,m,p=run(tr,te,'SVM','both')
    for key in ['verdict','fake_type','source_name','confidence','technique','source_type']:
        tab=collections.defaultdict(lambda:[0,0])
        for r,pp in zip(te,p):
            k=r[key] or 'NA'; tab[k][0]+=int(pp==r['label']); tab[k][1]+=1
        out[f'break_{lang}_{key}']={k:(c/n,n) for k,(c,n) in sorted(tab.items(),key=lambda x:-x[1][1])}
        print(lang,key,{k:f"{c/n:.2f}/{n}" for k,(c,n) in sorted(tab.items(),key=lambda x:-x[1][1]) if n>=5})
# temporal split: train <=2023, test >=2024 (on all splits pooled), per lang
for lang in ['ru','kk']:
    rs=[r for r in sel([lang]) if r['date']]
    tr=[r for r in rs if r['date'][:4]<='2023'];te=[r for r in rs if r['date'][:4]>='2024']
    print(lang,'temporal train',collections.Counter(r['label'] for r in tr),'test',collections.Counter(r['label'] for r in te))
    if lang=='kk':
        d,_,_=run(tr,te,'SVM','both'); out[f'temporal_{lang}']=d; print('temporal',lang,fmt(d))
# fine-grained fake type
from sklearn.metrics import classification_report
for lang in ['ru','kk','both']:
    L=['ru','kk'] if lang=='both' else [lang]
    cnt=collections.Counter(r['fake_type'] for r in sel(L,'train') if r['label']==1)
    keep={k for k,v in cnt.items() if v>=40}
    lab=lambda r: r['fake_type'] if r['fake_type'] in keep else 'Other'
    tr=[r for r in sel(L,'train') if r['label']==1];te=[r for r in sel(L,'test') if r['label']==1]
    m=model('SVM','both'); m.fit([r['text'] for r in tr],[lab(r) for r in tr]); p=m.predict([r['text'] for r in te]); y=[lab(r) for r in te]
    maj=collections.Counter(lab(r) for r in tr).most_common(1)[0][0]
    out[f'ftype_{lang}']=dict(classes=sorted(set(y)),f1m=f1_score(y,p,average='macro'),acc=accuracy_score(y,p),maj_acc=accuracy_score(y,[maj]*len(y)),n=len(y),report=classification_report(y,p,output_dict=True,zero_division=0))
    print('fake_type',lang,len(set(y)),'classes F1m=%.3f acc=%.3f majacc=%.3f n=%d'%(out[f'ftype_{lang}']['f1m'],out[f'ftype_{lang}']['acc'],out[f'ftype_{lang}']['maj_acc'],len(y)))
    # technique
    cnt=collections.Counter(r['technique'] for r in tr); keep2={k for k,v in cnt.items() if v>=40}
    lab2=lambda r: r['technique'] if r['technique'] in keep2 else 'Other'
    m=model('SVM','both'); m.fit([r['text'] for r in tr],[lab2(r) for r in tr]); p=m.predict([r['text'] for r in te]); y=[lab2(r) for r in te]
    maj=collections.Counter(lab2(r) for r in tr).most_common(1)[0][0]
    out[f'tech_{lang}']=dict(classes=sorted(set(y)),f1m=f1_score(y,p,average='macro'),acc=accuracy_score(y,p),maj_acc=accuracy_score(y,[maj]*len(y)),n=len(y))
    print('technique',lang,sorted(set(y)),'F1m=%.3f acc=%.3f majacc=%.3f'%(out[f'tech_{lang}']['f1m'],out[f'tech_{lang}']['acc'],out[f'tech_{lang}']['maj_acc']))
# learning curve
for lang in ['ru','kk']:
    tr=sel([lang],'train');te=sel([lang],'test');rr=np.random.default_rng(1)
    for frac in [0.1,0.25,0.5,1.0]:
        sc=[]
        for s in range(3 if frac<1 else 1):
            idx=rr.permutation(len(tr))[:int(frac*len(tr))]; d,_,_=run([tr[i] for i in idx],te,'SVM','both'); sc.append(d['f1m'])
        out[f'lc_{lang}_{frac}']=float(np.mean(sc)); print('lc',lang,frac,round(np.mean(sc),3))
json.dump(out,open('results/res_E4.json','w'),indent=1,default=float,ensure_ascii=False)
