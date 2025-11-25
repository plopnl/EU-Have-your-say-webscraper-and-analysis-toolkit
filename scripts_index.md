# 📑 Project Script Index

## 1️⃣ Data Collection  
### Group 1 — Fetching Scripts  
- `1_get_feedback_ids.py` → collect feedback IDs for later fetching and processing  
- `1_fetch_feedback_details_with_retry.py` → fetch raw feedback details with retry logic for robustness  
- `1_download_attachments.py` → download and store feedback attachments from the EU portal  
- `1_look_missing_ids.py` → check for missing feedback IDs and identify gaps in collection  

---

## 2️⃣ Preprocessing  
### Group 2 — Cleaning & Deduplication  
- `2_count_characters.py` → count characters in feedback entries for length analysis  
- `2_split_duplicate_unique.py` → split dataset into duplicate and unique feedback entries  
- `2_compare_attachment_texts.py` → compare attachment texts for duplication  
- `2_feedback_similarity_matrix.py` → compute similarity matrix for feedback entries  
- `2_attachments_similarity_matrix.py` → compute similarity matrix for attachments  

---

## 3️⃣ Translation / Similarity Analysis  
### Group 3 — Language Handling  
- `3_similarity_analysis.py` → analyze similarity among feedback entries  
- `3_similarity_analysis_attachments.py` → analyze similarity among attachments  

---

## 4️⃣ Bot Detection  
### Group 4 — Detection Scripts  
- `4_bot_detection_v2.py` → detect bot‑like submission patterns in feedback data (version 2 only)  

---

## 5️⃣ Exploratory Statistics  
### Group 5 — Deduplication & Frequency Analysis  
- `5_submission_volume_by_day.py` → visualize submission volume by day  
- `5_word_frequency_counter.py` → compute word frequency distributions  
- `5_bigrams.py` → extract and analyze bigram frequencies  
- `5_country.py` → summarize contributions by country  
- `5_keyword_counter.py` → count keyword occurrences in feedback  

---

## 6️⃣ Stance Detection & Evaluation  
### Group 6 — Analysis Scripts  
- `6_stance_detection.py` → main stance detection pipeline  
- `6_stance_analysis.py` → evaluate stance detection outputs against labeled data  
- `6_log_metrics.py` → log stance detection metrics and results  
- `6_manage_zero_shot.py` → manage zero‑shot stance detection runs  
- `6_reset_supervised.py` → reset supervised stance detection pipeline  

---

## 7️⃣ Feedback Visualizations  
### Group 7 — Visualization Scripts  
- `7_country_bar_chart.py` → visualize contributions by country with bar charts  
- `7_keyword_bar_chart.py` → visualize keyword frequencies with bar charts  
- `7_Length_Bin_Stance.py` → visualize stance distribution by feedback length bins  
- `7_plot_duplicates.py` → visualize duplicate contributions using similarity values and submission time differences  
- `7_split_by_usertype.py` → split feedback by userType and visualize submission categories  
