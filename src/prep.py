import json,re,collections,random,sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
D='data/raw/'
def ann(x):
    out={}
    a=x['annotations'][0] if x.get('annotations') else {'result':[]}
    for r in a['result']:
        v=r['value']
        if 'choices' in v: out[r['from_name']]=v['choices'][0]
        elif 'text' in v: out[r['from_name']]=' '.join(v['text'])
    return out
recs=[]
for f,lang in [('ru/real_ru.json','ru'),('ru/fake_ru.json','ru'),('kk/real_kz.json','kk'),('kk/fake_kz.json','kk')]:
    for x in json.load(open(D+f)):
        d=x['data'];a=ann(x)
        recs.append(dict(uid=None,src_id=x['id'],lang=lang,text=(d.get('final_text') or d['text']).strip(),title=d.get('title',''),
          label=1 if a.get('real_or_fake',d.get('system_label'))=='FAKE' else 0,
          source_name=d.get('source_name'),url=d.get('url',''),date=(d.get('date_news') or '')[:10],
          group_raw=d.get('split_group') or '',news_id=d.get('news_id'),verdict=d.get('factcheck_verdict') or '',
          source_type=a.get('source_type'),confidence=a.get('confidence'),credibility=a.get('source_credibility'),
          fake_type=a.get('fake_type'),technique=a.get('disinformation_technique'),
          system_label=d.get('system_label'),ann_label=a.get('real_or_fake')))
print('total',len(recs))
print('label disagreements system vs annot',sum((r['system_label'] or r['ann_label'])!=r['ann_label'] for r in recs))
# exact duplicates
norm=lambda t:re.sub(r'\W+',' ',t.lower()).strip()
seen={};dups=0;conf=0;keep=[]
for r in recs:
    k=(r['lang'],norm(r['text']))
    if k in seen:
        dups+=1; conf+= seen[k]['label']!=r['label']; continue
    seen[k]=r;keep.append(r)
print('exact dups removed',dups,'of which label-conflicting',conf)
recs=keep
# union-find groups
par=list(range(len(recs)))
def f(i):
    while par[i]!=i: par[i]=par[par[i]];i=par[i]
    return i
def u(i,j): par[f(i)]=f(j)
key=collections.defaultdict(list)
for i,r in enumerate(recs):
    if r['group_raw']: key[('g',r['lang'],r['group_raw'])].append(i)
    if r['url'] and 't.me' not in r['url']: key[('u',r['url'])].append(i)
for v in key.values():
    for j in v[1:]: u(v[0],j)
# near-duplicate via char tfidf cosine
for lang in ['ru','kk']:
    idx=[i for i,r in enumerate(recs) if r['lang']==lang]
    X=TfidfVectorizer(analyzer='char_wb',ngram_range=(4,5),min_df=2,sublinear_tf=True).fit_transform([recs[i]['text'] for i in idx])
    nn=NearestNeighbors(n_neighbors=6,metric='cosine').fit(X)
    dist,ind=nn.kneighbors(X)
    c=0
    for a in range(len(idx)):
        for d_,b in zip(dist[a][1:],ind[a][1:]):
            if 1-d_>=0.6: u(idx[a],idx[b]);c+=1
    print(lang,'near-dup links',c)
gid={};
for i,r in enumerate(recs): r['group']=f"G{f(i)}"
G=collections.defaultdict(list)
for r in recs: G[r['group']].append(r)
print('groups',len(G),'max size',max(len(v) for v in G.values()),'mixed-label groups',sum(len({x['label'] for x in v})>1 for v in G.values()))
# group split 70/15/15 stratified by majority label within lang
random.seed(13)
for lang in ['ru','kk']:
    gs=[g for g,v in G.items() if v[0]['lang']==lang]
    for lab in [0,1]:
        sub=[g for g in gs if round(np.mean([x['label'] for x in G[g]]))==lab]
        random.shuffle(sub);n=sum(len(G[g]) for g in sub);acc=0
        for g in sub:
            s='train' if acc<0.7*n else ('dev' if acc<0.85*n else 'test')
            for x in G[g]: x['split']=s
            acc+=len(G[g])
for r in recs:
    r['uid']=f"{r['lang']}_{'f' if r['label']==1 else 'r'}_{r.pop('src_id')}"
with open('data/corpus_base.jsonl','w') as w:
    for r in recs: w.write(json.dumps(r,ensure_ascii=False)+'\n')
c=collections.Counter((r['lang'],r['split'],r['label']) for r in recs)
for k in sorted(c): print(k,c[k])
