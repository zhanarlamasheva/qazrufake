"""Heuristic neutralisation of claim texts (RU/KK).

Removes (1) sentences that describe the distribution channel, the circulation
context, the speaker/venue context, fact-checker boilerplate or verdicts, and
(2) leading framing phrases ("Утверждается, что", "В публикациях говорится, что",
"якобы", "...деген ақпарат тарап жатыр"). If nothing is left, the first sentence
with phrase-level stripping is kept. Rule lists are reported in the paper appendix.
"""
import re

SPLIT = re.compile(r'(?<=[.!?…])\s+(?=[A-ZА-ЯЁӘІҢҒҮҰҚӨҺ«"“0-9])')

CHANNEL = (r"(whatsapp|telegram|телеграм|tiktok|тикток|instagram|инстаграм|facebook|фейсбук|youtube|ютуб|vk\b|вконтакте|twitter|threads|мессенджер\w*|соцсет\w*|социальн\w* сет\w*|сети интернет|в сети\b|интернет[еу]\b|пользовател\w*|блогер\w*|паблик\w*|канал\w*|видеоролик\w*|ролик\w*|пост\w*\b|публикаци\w*|в сми\b|ряде сми|рассылк\w*|"
           r"әлеуметтік желі\w*|желі\w*|желіде\w*|қолданушы\w*|пайдаланушы\w*|видеода\w*|видеоны\w*|роликте\w*|ролик\w*|жазба\w*|пост авторы\w*|парақша\w*|арна\w*)")
CIRCULATION = (r"(распространя\w+|распространил\w*|распространял\w*|разош\w+|разлетел\w*|циркулир\w*|вирусн\w*|репост\w*|тарат\w*|тарал\w*|тарады|тарап\w*|бөлісі\w*|жариялан\w*)")
FACTCHECK = (r"(фактчек\w*|factcheck|мы решили проверить|решили проверить|проверил\w* информаци\w*|подробнее читайте|доверяйте только|вердикт|не соответству\w+ действительност\w*|фейк\w*|опроверг\w*|ложн\w*|недостоверн\w*|манипуляц\w*|казинформ\s*-|"
             r"тексеру\w*|тексерд\w*|тексерейік|шындыққа сәйкес\w*|жалған\w*|фейк\w*|манипуляци\w*|ақиқат\w*|бұрмала\w*|үкім\w*)")
SPEAKER = (r"(\d{1,2}\s+(январ|феврал|март|апрел|ма[яй]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*.*\b(заявил|сказал|сообщил|отметил|рассказал|выступ)\w*|"
           r"\d{1,2}\s*-?\s*(қаңтар|ақпан|наурыз|сәуір|мамыр|маусым|шілде|тамыз|қыркүйек|қазан|қараша|желтоқсан)\w*.*(өткен|берген|сөйле|айтты|мәлімде|жазды|сұхбат)\w*|"
           r"отырысында|брифинг\w*|сұхбат\w*|интервью)")
SENT_DROP = re.compile(r"(" + CHANNEL + r".{0,80}" + CIRCULATION + r"|" + CIRCULATION + r".{0,80}" + CHANNEL + r"|" + FACTCHECK + r"|" + SPEAKER + r"|^сообщение распространялось)", re.I)

LEAD = [
    r"^(также\s+)?(в\s+(публикаци\w*|сообщени\w*|посте|видео\w*|ролике)\s+)?(также\s+)?(говорится|сообщается|утверждается|рассказывается|отмечается|указывается|обсуждается)\s*,?\s*(что|о том, что|о)?\s*",
    r"^(в\s+сети\s+)?распространя\w+\s+(информаци\w*|сообщени\w*|видео\w*)\s+(о том,?\s+что|о)\s*",
    r"^по (его|её|ее|их) словам,?\s*",
    r"^(депутаттың|спикердің|министрдің|оның|автордың)\s+айтуынша,?\s*",
]
WORD_DROP = re.compile(r"\b(якобы|будто бы|мол)\b\s*", re.I)
TAIL_KK = re.compile(r"\s*деген (ақпарат|хабарлама|мәлімет|пост|видео)\w*.*$", re.I)


def sentences(t):
    return [s.strip() for s in SPLIT.split(t.strip()) if s.strip()]


def strip_phrases(s):
    for p in LEAD:
        s = re.sub(p, '', s, flags=re.I)
    s = WORD_DROP.sub('', s)
    s = TAIL_KK.sub('.', s)
    s = s.strip(' ,;:-–—')
    return s[:1].upper() + s[1:] if s else s


def neutralize(t):
    ss = sentences(t)
    keep = [s for s in ss if not SENT_DROP.search(s)]
    if not keep:
        keep = ss[:1]
    out = ' '.join(strip_phrases(s) for s in keep)
    out = re.sub(r'\s+', ' ', out).strip()
    return out or t


# ---- claim-only extraction (used in the paper) ----
STOP = re.compile(r"(" + CHANNEL + r"|" + CIRCULATION + r"|" + FACTCHECK + r"|" + SPEAKER +
                  r"|на самом деле|в действительности|действительно|однако|тем не менее|напомним|ранее\b|кстати|"
                  r"шын мәнінде|алайда|дегенмен|бірақ|еске сала|негізінде|расында|шындығында)", re.I)
FRAME = re.compile(
    r"^.{0,120}?\b(распространя\w+|распространил\w*|публику\w+|пишут|сообща\w+|утвержда\w+|говорится|заявля\w+|тарап жатыр|таратып жатыр|жазып жатыр|айтылған|делінген)\b"
    r"\s*(сообщени\w*|информаци\w*|видео\w*|ролик\w*|пост\w*|новост\w*|слух\w*)?\s*,?\s*(о том,?\s*что|что|о)\s+", re.I)


def claim_only(t, max_sent=2):
    ss = sentences(t)
    out = []
    for i, s in enumerate(ss):
        if re.match(r"^сообщение распространялось", s, re.I):
            continue
        m = FRAME.match(s)
        if m and len(s) - m.end() > 20:          # framing sentence that embeds the claim
            s = s[m.end():]
        elif STOP.search(s):
            if out:
                break                              # context / rebuttal starts: stop
            continue                               # leading framing without claim: skip
        s = strip_phrases(s)
        if s:
            out.append(s)
        if len(out) >= max_sent:
            break
    if not out:
        out = [strip_phrases(ss[0])] if ss else [t]
    return re.sub(r"\s+", " ", ' '.join(out)).strip()
