# QazRuFake: a Kazakh–Russian benchmark of fact-checked claims and news posts

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24373377.svg)](https://doi.org/10.5281/zenodo.24373377)

QazRuFake is a corpus of **7,213 short texts in Russian (3,866) and Kazakh (3,347)** for fake news detection in Kazakhstan's bilingual information space. Its 3,133 FAKE items are claims checked by Kazakhstani fact-checking organisations and assigned to FAKE by documented corpus rules: Russian claims debunked by the outlet, and Kazakh Factcheck.kz claims with verdicts false, fake, mostly false or manipulation (505 of the 1,267 Kazakh FAKE items are manipulation verdicts). The repository also contains grouped train/dev/test splits, claim-only versions of all texts, model predictions and the code for every experiment in the accompanying paper.

> **Paper:** *Separating veracity from collection artefacts in fake news detection: a bilingual benchmark from Kazakhstan and an evaluation protocol* (under review). Citation details will be added on publication.

## Read this before using the data

- **The REAL class is heterogeneous.** It combines claims rated true by Factcheck.kz (Kazakh), news reports published by fact-checking outlets (Russian), and ordinary news posts (Tengrinews.kz Telegram, Kazakh-language outlets). Only the Kazakh Factcheck.kz items have both labels assigned by the same procedure. See [Label basis](#label-basis).
- **Texts are restatements, not original posts.** FAKE items are the fact-checker's summary of a claim, often with its circulation context. Models trained on them learn claim summaries, not raw social-media text.
- **Standard-split scores are inflated by collection artefacts.** In Russian, the source name alone gives macro-F1 0.905. Report the four-row evaluation protocol and QazRuFake-Core alongside the standard split (see [Evaluation protocol](#recommended-evaluations)).
- **Fine-grained labels have low inter-annotator agreement** (Cohen's κ 0.11–0.46). Treat `fake_type` and `technique` as exploratory, single-annotator metadata.

## Composition

| Lang. | Source | Text type | Label basis | Original verdict → label | Items |
| --- | --- | --- | --- | --- | --- |
| RU | Factcheck.kz, Kazinform, Aqparat Saqshysy, NoFake.kz | Claim summary | Debunked by the outlet | debunked → FAKE | 1,866 |
| RU | Same fact-checking outlets | News report | Publisher + manual review | – → REAL | 318 |
| RU | Tengrinews.kz (Telegram) | News post | Publisher | – → REAL | 1,682 |
| KK | Factcheck.kz (Kazakh) | Claim summary | Fact-checker verdict | false 494, fake 267, mostly false 1, manipulation 505 → FAKE | 1,267 |
| KK | Factcheck.kz (Kazakh) | Claim summary | Fact-checker verdict | true 816, mostly true 290, half true 199 → REAL | 1,305 |
| KK | Kazakh-language news outlets | News post | Publisher | – → REAL | 775 |

<a name="label-basis"></a>The **unambiguous subset** (`unamb = true`, 1,578 items) contains Kazakh Factcheck.kz claims with verdicts *true* vs *false*, *fake* or *mostly false*.

### Splits

Texts from the same fact-check article, with the same URL, or with character 4–5-gram cosine similarity ≥ 0.6 are grouped (6,036 groups), and whole groups are assigned to splits (70/15/15, stratified by language and label).

| Language | Train (REAL / FAKE) | Dev | Test |
| --- | --- | --- | --- |
| Russian | 1,400 / 1,307 | 300 / 280 | 300 / 279 |
| Kazakh | 1,467 / 879 | 300 / 199 | 313 / 189 |

## Files

```
data/
  qazrufake.jsonl          # the corpus: one JSON object per line (fields below)
  raw/ru, raw/kk           # original Label Studio exports and labelling config
annotation/
  double_annotation_295.csv            # linguist vs fact-checker labels on 295 FAKE items
  double_annotation_sample.xlsx, first_annotation_key.csv, agreement.py (κ from the CSV)
predictions/
  svm_wordchar_predictions.json        # per-item SVM predictions (test sets)
  xlmr_base_predictions.jsonl          # per-item XLM-R probabilities, 9 settings x 3 seeds
  xlmr_v3_predictions.jsonl            # XLM-R trained on Russian fact-checking outlets, 3 seeds
  llm_zeroshot_predictions.jsonl       # zero-shot Qwen2.5-7B-Instruct, English prompt, P(FAKE) per test item
  xlmr_v4_classweighted_predictions.jsonl   # XLM-R, class-weighted loss, protocol rows, 5-10 seeds
  llm_v4_qwen14b_native_predictions.jsonl   # zero-shot Qwen2.5-14B-Instruct, prompt in the text's language, dev + test
  labse_embeddings.npz                 # LaBSE embeddings of all texts (duplicate checks)
results/                   # every number reported in the paper, as JSON
figures/                   # paper figures
src/                       # corpus construction and all experiments
notebooks/                 # Colab notebooks for transformer fine-tuning (GPU)
```

### Fields in `data/qazrufake.jsonl`

| Field | Description |
| --- | --- |
| `uid` | Unique id: `{lang}_{r|f}_{Label Studio task id}` |
| `lang` | `ru` or `kk` |
| `text` | Original text (claim summary or news post) |
| `claim` | Claim-only version: channel, framing, speaker context and rebuttals removed by rule (`src/neutralize.py`) |
| `label` | 1 = FAKE, 0 = REAL |
| `split` | `train` / `dev` / `test` |
| `group` | Group id used for splitting (keep groups together in any new split) |
| `source_name` | Source outlet |
| `verdict` | Original fact-checker verdict (Kazakh Factcheck.kz items only) |
| `fc_kk`, `unamb` | Kazakh Factcheck.kz item; unambiguous-verdict subset |
| `fake_type`, `technique` | Fine-grained labels for FAKE items (13 and 7 categories; low agreement, see above) |
| `source_type`, `confidence`, `credibility` | Annotation metadata. For Kazakh fact-check items, `confidence` encodes the verdict strength, not annotator certainty |
| `url`, `date`, `title`, `news_id`, `group_raw` | Provenance |
| `system_label`, `ann_label` | Labels as stored in the Label Studio export |

## Main results

Macro-F1 on the test split. SVM: linear SVM over word 1–2-grams and character 2–5-grams. XLM-R: base model, mean ± SD over three seeds.

| Evaluation | SVM | XLM-R |
| --- | --- | --- |
| Russian, standard split | 0.955 | 0.967 ± 0.005 |
| Russian, source name only (metadata probe) | 0.905 | – |
| Russian, claim-only text | 0.877 | 0.899 ± 0.004 |
| Kazakh, standard split | 0.824 | 0.830 ± 0.005 |
| Kazakh Factcheck.kz, unambiguous verdicts | 0.846 | 0.850 ± 0.010 |
| Kazakh Factcheck.kz, unambiguous verdicts, claim-only | 0.814 | 0.831 ± 0.003 |
| Zero-shot RU → KK / KK → RU (char n-gram SVM) | 0.442 / 0.468 | 0.766 / 0.782 |

<a name="recommended-evaluations"></a>**Evaluation protocol.** Report these four rows alongside any headline score (paper, Table 12), and QazRuFake-Core as the headline evaluation. XLM-R: class-weighted loss, mean over 5 seeds (10 for row 4 RU and Core). LLM: Qwen2.5-14B-Instruct, prompt in the language of the text, threshold chosen on dev.

| Row | What it removes | SVM (RU / KK) | XLM-R (RU / KK) | Zero-shot LLM (RU / KK) |
| --- | --- | --- | --- | --- |
| 1. Standard split | nothing (reference) | 0.955 / 0.824 | 0.965 / 0.831 | 0.808 / 0.726 |
| 2. Metadata-only probe (best of source name, source type, date) | the text | 0.905 / 0.755 | – | – |
| 3. Same publisher (train and test only on publishers with both labels) | publisher and register differences | 0.825 / 0.811 | 0.913 / 0.804 | 0.663 / 0.707 |
| 4. Claim-only text, within row 3 | fact-checker framing, channel and rebuttal cues | 0.694 / 0.738 | 0.722 / 0.777 | 0.674 / 0.679 |
| **QazRuFake-Core**: Kazakh Factcheck.kz, unambiguous verdicts (`unamb`), `claim` text | graded verdicts | – / 0.814 | – / 0.824 | – / 0.757 |

Use group-level bootstrap for confidence intervals, and report ROC-AUC next to macro-F1 when a row changes the class balance. The Kazakh temporal split (train ≤ 2023, test ≥ 2024) is a useful robustness check. Controls in the paper (Table 9): balancing the training set leaves the Russian drop unchanged (ROC-AUC 0.988 → 0.961 → 0.854), while cutting original texts to claim-only length explains part of the claim-only loss in Russian and all of it in Kazakh. Without class weights, one of three XLM-R seeds collapsed on row 4 RU (0.650 ± 0.177); see `results/res_v4.json` and the paper's Appendix A.

**External corpus.** `src/external_fnn.py` applies the protocol to FakeNewsNet titles (PolitiFact, GossipCop; CSVs from github.com/KaiDMML/FakeNewsNet): the publisher domain alone gives 0.819 vs 0.859 for a title SVM on PolitiFact, while GossipCop keeps its score under same-publisher evaluation (0.769 vs 0.778).

## Reproducing

```bash
pip install -r requirements.txt
python src/prep.py            # raw exports -> data/corpus_base.jsonl (dedup, groups, splits)
python src/make_corpus.py     # -> data/qazrufake.jsonl (adds claim-only text, subset flags)
python src/exp.py             # Tables 4-5 (n-gram baselines, cross-lingual)
python src/exp2.py            # metadata probes, cue masking, within-source
python src/exp3.py            # per-verdict errors, fine-grained typing, learning curves
python src/stats.py           # corpus statistics and figures
python src/xling.py           # anchor-based bilingual duplicate candidates
python src/exp_rev.py         # claim-only, unambiguous, temporal, duplicate sensitivity; SVM predictions
python src/analyze_v2.py predictions/xlmr_base_predictions.jsonl predictions/labse_embeddings.npz
python src/analyze_v2_extra.py
python src/analyze_v3.py predictions/xlmr_v3_predictions.jsonl predictions/llm_zeroshot_predictions.jsonl
python src/analyze_r3.py      # ROC-AUC, balanced-test, class-balance, training-size and length controls (Table 9)
python src/external_fnn.py <dir with FakeNewsNet CSVs>   # protocol on FakeNewsNet (Table 11)
python src/analyze_v4.py      # class-weighted XLM-R and Qwen2.5-14B reruns (Tables 7, 12, A2)
python src/fig_protocol.py     # Figure 1 (evaluation protocol)
```

All scripts run from the repository root. `prep.py` + `make_corpus.py` reproduce `data/qazrufake.jsonl` exactly. Transformer fine-tuning needs a GPU: open `notebooks/qazrufake_v2.ipynb` (and `qazrufake_v3.ipynb` for the same-publisher XLM-R rows and the zero-shot LLM) in Google Colab (T4), upload `data/qazrufake.jsonl` and run all cells.

## Licence

- **Annotations, splits, derived data and predictions:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (see `LICENSE-DATA`).
- **Code:** MIT (see `LICENSE`).
- **Texts** remain the property of their original publishers (Factcheck.kz, Kazinform, Aqparat Saqshysy, NoFake.kz, Tengrinews.kz and the Kazakh-language outlets listed in the data). They are redistributed here for non-commercial research only. If you are a rights holder and want material removed, please open an issue.

## Ethics

FAKE items repeat false claims, including health-related ones. Do not republish them without their debunking context. Detectors trained on this corpus are meant to assist human fact-checkers, not to make automated moderation decisions: their errors fall disproportionately on partly true content.

## Citation

Until the paper is published, please cite the dataset:

```bibtex
@dataset{lamasheva_2026_qazrufake,
  author    = {Lamasheva, Zhanar},
  title     = {{QazRuFake: a Kazakh-Russian benchmark of fact-checked claims and news posts for fake news detection}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24373377},
  url       = {https://doi.org/10.5281/zenodo.24373377}
}
```
