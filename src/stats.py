from lib import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
st={}
for lang in ['ru','kk']:
    for lab in [0,1]:
        rs=[r for r in R if r['lang']==lang and r['label']==lab]
        w=[len(r['text'].split()) for r in rs]
        st[f'{lang}_{lab}']=dict(n=len(rs),words_mean=np.mean(w),words_med=float(np.median(w)),tokens=int(sum(w)),
            sources=collections.Counter(r['source_name'] for r in rs).most_common(),
            years=(min(r['date'][:4] for r in rs if r['date']),max(r['date'][:4] for r in rs if r['date'])),
            undated=sum(not r['date'] for r in rs), vocab=len({t for r in rs for t in re.findall(r'\w+',r['text'].lower())}),
            groups=len({r['group'] for r in rs}))
        print(lang,lab,{k:v for k,v in st[f'{lang}_{lab}'].items() if k!='sources'}, st[f'{lang}_{lab}']['sources'][:6])
st['groups_total']=len({r['group'] for r in R}); st['n_total']=len(R)
st['ftype']={lang:collections.Counter(r['fake_type'] for r in R if r['lang']==lang and r['label']==1).most_common() for lang in ['ru','kk']}
st['tech']={lang:collections.Counter(r['technique'] for r in R if r['lang']==lang and r['label']==1).most_common() for lang in ['ru','kk']}
st['verdict']=collections.Counter((r['verdict'],r['label']) for r in R if r['lang']=='kk').most_common()
json.dump(st,open('results/stats.json','w'),ensure_ascii=False,indent=1,default=float)
print(st['groups_total'],st['n_total'])
# Fig 1: year distribution
fig,ax=plt.subplots(1,2,figsize=(10,3.2),sharey=False)
yrs=[str(y) for y in range(2017,2027)]
for a,lang,name in zip(ax,['ru','kk'],['Russian','Kazakh']):
    for lab,c,off,nm in [(0,'#4C72B0',-0.2,'Real'),(1,'#C44E52',0.2,'Fake')]:
        cnt=collections.Counter(r['date'][:4] for r in R if r['lang']==lang and r['label']==lab)
        a.bar(np.arange(len(yrs))+off,[cnt.get(y,0) for y in yrs],0.4,color=c,label=nm)
    a.set_xticks(range(len(yrs)));a.set_xticklabels(yrs,rotation=45,fontsize=8);a.set_title(name+(' (776 undated real items omitted)' if lang=='kk' else ''),fontsize=9)
    a.spines[['top','right']].set_visible(False)
ax[0].set_ylabel('Items');ax[0].legend(frameon=False)
plt.tight_layout();plt.savefig('figures/fig1_years.png',dpi=200)
# Fig 2: shortcut comparison
E1=json.load(open('results/res_E1E2.json'));E3=json.load(open('results/res_E3.json'));E4=json.load(open('results/res_E4.json'))
labels=['Text\n(word+char SVM)','Source name\nonly','Source type\nonly','Year\nonly','Surface\nfeatures','Text, mixed-\nlabel sources']
vals={'ru':[E1['E1_ru_SVM_both']['f1m'],E3['meta_ru_source_name']['f1m'],E3['meta_ru_source_type']['f1m'],E3['meta_ru_year']['f1m'],E3['meta_ru_surface']['f1m'],E3['within_ru']['f1m']],
      'kk':[E1['E1_kk_SVM_both']['f1m'],E3['meta_kk_source_name']['f1m'],E3['meta_kk_source_type']['f1m'],E3['meta_kk_year']['f1m'],E3['meta_kk_surface']['f1m'],E3['within_kk']['f1m']]}
fig,a=plt.subplots(figsize=(9,3.4));x=np.arange(len(labels))
a.bar(x-0.2,vals['ru'],0.4,label='Russian',color='#4C72B0');a.bar(x+0.2,vals['kk'],0.4,label='Kazakh',color='#DD8452')
for i in range(len(x)):
    a.text(x[i]-0.2,vals['ru'][i]+0.01,f"{vals['ru'][i]:.2f}",ha='center',fontsize=7);a.text(x[i]+0.2,vals['kk'][i]+0.01,f"{vals['kk'][i]:.2f}",ha='center',fontsize=7)
a.set_xticks(x);a.set_xticklabels(labels,fontsize=8);a.set_ylabel('Macro-F1 (test)');a.set_ylim(0,1.05);a.legend(frameon=False,fontsize=8);a.spines[['top','right']].set_visible(False)
plt.tight_layout();plt.savefig('figures/fig2_shortcuts.png',dpi=200)
# Fig 3: learning curve
fig,a=plt.subplots(figsize=(4.5,3.2));fr=[0.1,0.25,0.5,1.0]
for lang,nm,c in [('ru','Russian','#4C72B0'),('kk','Kazakh','#DD8452')]:
    a.plot([f*100 for f in fr],[E4[f'lc_{lang}_{f}'] for f in fr],'o-',color=c,label=nm)
a.set_xlabel('% of training data');a.set_ylabel('Macro-F1 (test)');a.legend(frameon=False);a.spines[['top','right']].set_visible(False)
plt.tight_layout();plt.savefig('figures/fig3_lc.png',dpi=200)
