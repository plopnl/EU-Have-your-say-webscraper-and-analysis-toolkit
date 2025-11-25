# EU Have Your Say — Consultation Analysis Toolkit

A research pipeline for scraping, classifying, and analysing public feedback from the European Commission's [Have Your Say](https://ec.europa.eu/info/law/better-regulation/have-your-say) portal. Designed for transparent, reproducible public-interest analysis.

---

## What it does

1. **Fetches** all submissions for a given consultation via the EU HYS public API
2. **Deduplicates** near-identical submissions (TF-IDF + cosine similarity, 0.95 threshold)
3. **Classifies stance** (for / against / unclear) using a three-layer hybrid model that improves iteratively through active learning
4. **Detects coordinated submissions** using semantic embeddings and burst-timing analysis
5. **Analyses content** — keywords, bigrams, volume over time, country and language breakdown
6. **Visualises** everything as publication-quality matplotlib figures

---

## Stack

| Layer | Technology |
|---|---|
| Data collection | Python, `requests`, EU HYS REST API (public, no key required) |
| NLP / stance | `transformers` (DeBERTa-v3-large), `sentence-transformers` |
| Stance labeling | Claude API (Anthropic) — primary label oracle |
| Bot detection | `distiluse-base-multilingual-cased-v2`, cosine similarity |
| Visualisation | `matplotlib`, `seaborn` |

---

## Quick start

```bash
# 1. Clone and create environment
git clone https://github.com/plopnl/EU-Have-your-say-webscraper-and-analysis-toolkit
cd EU-Have-your-say-webscraper-and-analysis-toolkit
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure the consultation (edit settings.py)
#    Set: DEFAULT_PUBLICATION_ID, INITIATIVE_ID, INITIATIVE_SLUG,
#         CONSULTATION_START, CONSULTATION_END

# 3. Run the interactive pipeline menu
./run_pipeline.sh
#    1 → fetch raw data from the EU API
#    9 → run the full pipeline end to end
#    a → rerun stance detection + metrics
```

All scripts must be run from the project root. `data/` and `figures/` are created automatically.

> **GPU note:** Stage 3 supervised training benefits strongly from a CUDA GPU with bfloat16 support (Ampere+). Stage 5 bot detection builds an N×N similarity matrix — becomes slow above ~50k entries without a GPU.

---

## Pipeline stages

| # | Scripts | Key output |
|---|---|---|
| 1 | `1_get_feedback_ids.py`, `1_fetch_feedback_details_with_retry.py`, `1_download_attachments.py` | `data/feedback_details.csv` |
| 2 | `2_split_duplicate_unique.py`, `2_feedback_similarity_matrix.py`, … | Cleaned CSVs, similarity matrices |
| 3 | `3_stance_detection.py`, `3_log_metrics.py` | `data/stance_detection/stance_supervised.csv`, trained model checkpoint |
| 4 | `4_similarity_analysis.py`, `4_similarity_analysis_attachments.py` | Top-N similarity CSVs for search |
| 5 | `5_bot_detection_v4.py`, `5_bot_detection_v4_analysis.py` | `data/bot_detection_v4.csv` |
| 6 | `6_word_frequency_counter.py`, `6_bigrams.py`, `6_country.py`, … | Keyword / bigram / country CSVs |
| 7 | `7_stance_analysis.py`, `7_stance_by_usertype.py`, … | `figures/` stance plots |
| 8 | `8_country_bar_chart.py`, `8_submission_timing.py`, … | `figures/` all remaining plots |

See [`scripts_index.md`](scripts_index.md) for per-script descriptions.

---

## Stance detection in detail

Four layers in total, each toggled by a boolean in `3_stance_detection.py`:

**1. Rule-based** — keyword lists with word-boundary matching and a scoring system (both sides scored, winner must lead by ≥1 point). Fast but no semantic understanding.

**2. Claude API** — sends submissions to a Claude model with a consultation-specific system prompt. Uses prompt caching for cost efficiency. Results stored in `claude_labels.csv`; incremental runs append new entries only. This is now the primary label oracle for supervised training.

**3. Zero-shot** — three DeBERTa-based models run in parallel. When all three agree, their consensus label is used as a fallback where Claude labels are absent. Run once; results are cached.

**4. Supervised** — fine-tunes `microsoft/deberta-v3-large` on the best available labels. Improves each run as new manual reviews accumulate.

Label priority (highest → lowest): **manual review** → **Claude** → ZS consensus → pseudo-labels (≥0.85 confidence) → rule-based.

Design principle: *unclear is better than wrong* — no negation logic, no keyword overrides; only manual labels and model consensus flip a result.

---

## Reusing for a new consultation

```bash
# 1. Archive current consultation artefacts
mv data/stance_detection/supervised_model   data/stance_detection/supervised_model_<slug>
mv data/stance_detection/stance_zero_shot_*.csv  <archive>/
mv data/stance_detection/manual_review.csv  <archive>/

# 2. Update settings.py
DEFAULT_PUBLICATION_ID  = <new id>       # from the EU HYS URL
INITIATIVE_ID           = <new id>
INITIATIVE_SLUG         = "<new slug>"
CONSULTATION_START      = "<DD Month YYYY>"
CONSULTATION_END        = "<DD Month YYYY>"
```

Everything else reuses automatically: all scripts and the base model (`microsoft/deberta-v3-large`) are consultation-agnostic.

---

## Configuration

Copy `settings.py` — it centralises all tunable parameters. The most commonly changed per-consultation:

| Setting | Default | Notes |
|---|---|---|
| `DEFAULT_PUBLICATION_ID` | — | From EU HYS URL |
| `DUPLICATE_THRESHOLD` | 0.95 | TF-IDF cosine threshold for near-duplicates |
| `BURST_GAP_SECONDS` | 60 | Bot detection burst window |
| `PSEUDO_CONF_THRESHOLD` | 0.85 | Min confidence to accept a pseudo-label |
| `USE_BF16` | True | Set False if GPU doesn't support bfloat16 |

---

## Ethical note

This toolkit is intended for **research and public-interest analysis** of publicly available EU consultation data. Results are indicative — stance classifications are model-generated and may contain errors. Do not treat automated classifications as definitive policy conclusions. The EU portal's terms of service apply to any data you collect.

---

## Contributing

Issues and PRs welcome. See [`scripts_index.md`](scripts_index.md) for the full script reference.
