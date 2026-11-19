"""Anchor-based candidate pairs of Russian-Kazakh duplicates (numbers, Latin tokens,
transliterated proper names), IDF-weighted cosine over shared anchors (>=2 shared)."""
import json, re, collections, math
R = [json.loads(l) for l in open('data/qazrufake.jsonl', encoding='utf-8')]
TR = str.maketrans({'ә': 'а', 'і': 'и', 'ң': 'н', 'ғ': 'г', 'ү': 'у', 'ұ': 'у', 'қ': 'к', 'ө': 'о', 'һ': 'х', 'ё': 'е', 'й': 'и'})
STOPCAP = {'в', 'на', 'по', 'и', 'с', 'о', 'что', 'это', 'бұл', 'және', 'мен', 'деп', 'также', 'однако', 'при', 'для', 'из', 'к', 'у', 'от'}
YEARS = {'1', '2', '3', '4', '5', '10', '2020', '2021', '2022', '2023', '2024', '2025', '2026'}

def anchors(t):
    a = set()
    t2 = re.sub(r'(?<=\d)[\s ](?=\d{3}\b)', '', t)
    for n in re.findall(r'\d+(?:[.,]\d+)?', t2):
        n = n.replace(',', '.')
        if n not in YEARS and len(n) >= 2: a.add('N' + n)
    for w in re.findall(r'[A-Za-z][A-Za-z\-]{2,}', t): a.add('L' + w.lower())
    for w in re.findall(r'(?<![.!?]\s)(?<!^)\b[А-ЯЁӘІҢҒҮҰҚӨҺ][а-яёәіңғүұқөһ]{3,}', t):
        w = w.lower().translate(TR)
        if w not in STOPCAP: a.add('C' + w[:5])
    return a

A = {r['uid']: anchors(r['text']) for r in R}
df = collections.Counter(x for s in A.values() for x in s)
idf = lambda x: math.log(len(R) / df[x])
ru = [r for r in R if r['lang'] == 'ru']; kk = [r for r in R if r['lang'] == 'kk']
inv = collections.defaultdict(list)
for r in kk:
    for x in A[r['uid']]: inv[x].append(r['uid'])
pairs = []
for r in ru:
    cand = collections.Counter()
    for x in A[r['uid']]:
        if df[x] > 30: continue
        for u in inv[x]: cand[u] += 1
    na = math.sqrt(sum(idf(x) ** 2 for x in A[r['uid']])) or 1
    for u, c in cand.items():
        if c < 2: continue
        sh = A[r['uid']] & A[u]
        s = sum(idf(x) ** 2 for x in sh) / (na * (math.sqrt(sum(idf(x) ** 2 for x in A[u])) or 1))
        pairs.append((s, sorted(sh), r['uid'], u))
pairs.sort(key=lambda p: -p[0])
json.dump(pairs, open('results/xling_pairs.json', 'w'), ensure_ascii=False)
print(len(pairs), 'candidate pairs')
