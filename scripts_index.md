# Project Script Index

## Stage 1 — Data Collection

- `1_get_feedback_ids.py` — collect feedback IDs from the EU Have Your Say API
- `1_fetch_feedback_details_with_retry.py` — fetch full feedback details with retry logic
- `1_download_attachments.py` — download attachments referenced in feedback submissions
- `1_look_missing_ids.py` — identify gaps in collected feedback IDs

---

## Stage 2 — Preprocessing

- `2_split_duplicate_unique.py` — cluster near-duplicate submissions (TF-IDF + cosine similarity)
- `2_count_characters.py` — compute word and character counts per submission
- `2_feedback_similarity_matrix.py` — full pairwise similarity matrix for text feedback
- `2_attachments_similarity_matrix.py` — pairwise similarity matrix for attachment texts
- `2_compare_attachment_texts.py` — compare attachment texts across submissions
- `2_translate_attachments.py` — translate attachment text to English

---

## Stage 3 — Stance Detection

- `3_stance_detection.py` — main stance detection pipeline (rule-based → zero-shot → supervised fine-tuning)
- `3_similarity_analysis.py` — semantic similarity analysis of feedback text
- `3_similarity_analysis_attachments.py` — semantic similarity analysis of attachment text
- `3_log_metrics.py` — log stance detection run metrics
- `3_manage_zero_shot.py` — manage zero-shot model runs
- `3_reset_supervised.py` — reset supervised model to start fresh from base model

---

## Stage 4 — Bot Detection (v2/v3)

- `4_bot_detection_v2.py` — bot detection using burst timing analysis (v2)
- `4_bot_detection_v3.py` — bot detection with improved burst detection (v3)
- `4_similarity_analysis.py` — similarity analysis used in bot detection pipeline
- `4_similarity_analysis_attachments.py` — attachment similarity for bot detection

---

## Stage 5 — Bot Detection (v4) & Content Analysis

- `5_bot_detection_v4.py` — semantic embedding-based bot/coordination detection; outputs `data/bot_detection_v4.csv`
- `5_bot_detection_v4_analysis.py` — visualize bot detection results
- `5_bot_detection_v4.1_analysis.py` — extended analysis of v4 results
- `5_submission_volume_by_day.py` — submission volume over time
- `5_keyword_counter.py` — keyword frequency counts
- `5_word_frequency_counter.py` — word frequency distributions
- `5_country.py` — submission counts by country
- `5_bigrams.py` — bigram frequency extraction

---

## Stage 6 — Analysis

- `6_stance_detection.py` — stance detection (updated pipeline version)
- `6_stance_detection_v1.py` — stance detection v1 (reference)
- `6_stance_analysis.py` — evaluate and summarize stance detection outputs
- `6_bot_detection_v4_analysis.py` — bot detection analysis (stage 6 version)
- `6_submission_volume_by_day.py` — submission volume figures
- `6_keyword_counter.py` — keyword counts
- `6_word_frequency_counter.py` — word frequency
- `6_country.py` — country breakdown
- `6_bigrams.py` — bigram extraction
- `6_log_metrics.py` — log model run metrics
- `6_manage_zero_shot.py` — zero-shot model management
- `6_reset_supervised.py` — reset supervised model

---

## Stage 7 — Stance Visualizations

- `7_stance_analysis.py` — stance distribution plots
- `7_stance_by_usertype.py` — stance by submitter type
- `7_stance_by_usertype_dedup.py` — stance by usertype with deduplication weighting
- `7_country_bar_chart.py` — country contribution bar charts
- `7_keyword_bar_chart.py` — keyword frequency bar charts
- `7_split_by_usertype.py` — split feedback by userType
- `7_plot_duplicates.py` — visualize duplicate clusters
- `7_Length_Bin_Stance.py` — stance distribution by feedback length

---

## Stage 8 — Final Visualizations

- `8_country_bar_chart.py` — country bar charts (final)
- `8_keyword_bar_chart.py` — keyword bar charts (final)
- `8_split_by_usertype.py` — usertype split figures (final)
- `8_plot_duplicates.py` — duplicate visualization (final)
- `8_Length_Bin_Stance.py` — length bin stance plots (final)
- `8_submission_timing.py` — submission timing heatmaps by day × hour

---

## Shared Configuration

- `settings.py` — centralised settings: file paths, publication metadata, analysis thresholds, plot styling
- `rules_base.json` — keyword rules for rule-based stance detection
- `run_pipeline.sh` — interactive menu to run pipeline stages in order
