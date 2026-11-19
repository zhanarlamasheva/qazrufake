"""Adds claim-only text and subset flags to data/corpus_base.jsonl -> data/qazrufake.jsonl."""
import json
from neutralize import claim_only
with open('data/qazrufake.jsonl', 'w', encoding='utf-8') as out:
    for l in open('data/corpus_base.jsonl', encoding='utf-8'):
        r = json.loads(l)
        r['claim'] = claim_only(r['text'])
        r['fc_kk'] = r['lang'] == 'kk' and r['source_name'] == 'FactcheckKz_kz'
        r['unamb'] = r['fc_kk'] and (r['verdict'] == 'true' or r['verdict'] in ('false', 'fake', 'mostly-false'))
        out.write(json.dumps(r, ensure_ascii=False) + '\n')
