# ============================================
# 📄 EU Feedback Attachment Deduplication Script
# Created by Plop (Twitter/GitHub: @plopnl)
# Description: Processes consultation feedback attachments from the EU "Have Your Say" portal.
#              Extracts text from PDF, DOCX, ODT, and TXT files, normalizes content,
#              and compares documents using TF-IDF + cosine similarity.
#              Flags near-duplicate attachments, groups them into clusters,
#              and separates unique files. Broken/unreadable files are moved aside.
# Source: https://ec.europa.eu/info/law/better-regulation/have-your-say
# ============================================

import os
import re
import hashlib
import pandas as pd
import pdfplumber
import zipfile
import shutil
from docx import Document
from odf import text, teletype
from odf.opendocument import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# =========================
# 🔧 SETTINGS
# =========================
ATTACHMENT_DIR = "attachments"
SIMILARITY_THRESHOLD = 0.95
BROKEN_DIR = os.path.join(ATTACHMENT_DIR, "broken")
os.makedirs(BROKEN_DIR, exist_ok=True)

# =========================
# 🆔 FEEDBACK ID EXTRACTION
# =========================
def extract_feedbackId(filename: str) -> str:
    m = re.match(r"^\s*(\d+)(?:[_\s].*)?$", filename)
    if m:
        return m.group(1)
    return hashlib.sha1(filename.encode("utf-8")).hexdigest()[:16]

# =========================
# 🧹 CLEAN FILENAME (for plotting/labels)
# =========================
def clean_filename(name: str) -> str:
    name = re.sub(r"^\s*\d+[_\s]+", "", name)
    name = os.path.splitext(name)[0]
    return name
    
# =========================
# 🧹 CLEAN text from attachments for safe storage
# =========================
def clean_attachment_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove ARES-style headers
    text = re.sub(
        #r"ref\.?\s*ares\(\d+\)\s*[-–:]\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        r"ref\.?\s*ares\(\d+\)\s*\d+\s*[-–—:]\s*\d{1,4}[\/.-]\d{1,2}[\/.-]\d{1,4}\b",
        "",
        text,
        flags=re.IGNORECASE
    )
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================
# 📄 TEXT EXTRACTION
# =========================
def extract_text(filepath):
    try:
        if filepath.lower().endswith(".pdf"):
            with pdfplumber.open(filepath) as pdf:
                return "\n".join([page.extract_text() or "" for page in pdf.pages])
        elif filepath.lower().endswith(".docx"):
            if not zipfile.is_zipfile(filepath):
                raise ValueError("Not a valid DOCX file")
            doc = Document(filepath)
            return "\n".join([para.text for para in doc.paragraphs])
        elif filepath.lower().endswith(".odt"):
            odt_doc = load(filepath)
            paragraphs = odt_doc.getElementsByType(text.P)
            return "\n".join([teletype.extractText(p) for p in paragraphs])
        elif filepath.lower().endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        print(f"❌ Failed to read {filepath}: {e}")
        try:
            shutil.move(filepath, os.path.join(BROKEN_DIR, os.path.basename(filepath)))
            print(f"📁 Moved to broken/: {os.path.basename(filepath)}")
        except Exception as move_err:
            print(f"⚠️ Failed to move broken file: {move_err}")
    return ""

# =========================
# 📥 LOAD & NORMALIZE TEXTS
# =========================
files, feedback_ids, texts, char_counts, word_counts = [], [], [], [], []

for filename in os.listdir(ATTACHMENT_DIR):
    path = os.path.join(ATTACHMENT_DIR, filename)
    if os.path.isfile(path) and filename.lower().endswith((".pdf", ".docx", ".odt", ".txt")):
        content = extract_text(path).strip().lower()
        if content:
            files.append(filename)
            feedback_ids.append(extract_feedbackId(filename))
            texts.append(content)
            char_counts.append(len(content))
            word_counts.append(len(content.split()))

if not files:
    print("⚠️ No readable attachments found. Exiting.")
    pd.DataFrame(columns=[
        "feedbackId","filename","clean_filename","group_id","group_size",
        "matched_ids","matched_files","max_similarity_in_group","similarity_scores",
        "char_count","word_count"
    ]).to_csv("data/attachment_duplicates.csv", index=False)
    pd.DataFrame(columns=[
        "feedbackId","filename","clean_filename","char_count","word_count"
    ]).to_csv("data/attachment_unique.csv", index=False)
    raise SystemExit(0)

# =========================
# 🔍 VECTORIZE & COMPARE
# =========================
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(texts)
similarity_matrix = cosine_similarity(tfidf_matrix)

# =========================
# 📊 GROUP DETECTION
# =========================
def detect_attachment_groups(files, feedback_ids, similarity_matrix, threshold, char_counts, word_counts):
    graph = defaultdict(set)
    sim_scores = defaultdict(list)

    n = len(files)
    for i in range(n):
        for j in range(n):
            if i != j and similarity_matrix[i, j] > threshold:
                graph[files[i]].add(files[j])
                graph[files[j]].add(files[i])
                sim_scores[files[i]].append(similarity_matrix[i, j])
                sim_scores[files[j]].append(similarity_matrix[i, j])

    visited = set()
    groups = []
    for node in graph:
        if node not in visited:
            stack = [node]
            group = set()
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    group.add(current)
                    stack.extend(graph[current] - visited)
            groups.append(group)

    idx_by_file = {fname: k for k, fname in enumerate(files)}

    rows = []
    for group_id, group in enumerate(groups):
        group_size = len(group)
        group_ids = [feedback_ids[idx_by_file[f]] for f in group]
        group_filenames = [clean_filename(f) for f in group]

        for fname in group:
            k = idx_by_file[fname]
            fid = feedback_ids[k]
            scores = sim_scores.get(fname, [])
            max_sim = max(scores, default=None)

            rows.append({
                "feedbackId": fid,
                "filename": fname,
                "clean_filename": clean_filename(fname),
                "group_id": group_id,
                "group_size": group_size,
                "matched_ids": "; ".join(sorted(group_ids)),
                "matched_files": "; ".join(sorted(group_filenames)),
                "max_similarity_in_group": round(max_sim, 4) if max_sim is not None else None,
                "similarity_scores": "; ".join(f"{s:.4f}" for s in scores),
                "char_count": char_counts[k],
                "word_count": word_counts[k]
            })

    return pd.DataFrame(rows).reset_index(drop=True)

# =========================
# 💾 SAVE RESULTS
# =========================
df_duplicates = detect_attachment_groups(files, feedback_ids, similarity_matrix,
                                         SIMILARITY_THRESHOLD, char_counts, word_counts)

# Add text column for duplicates
df_duplicates["attachment_text"] = [
    clean_attachment_text(texts[files.index(f)]) for f in df_duplicates["filename"]
]

df_duplicates.to_csv("data/attachment_duplicates.csv", index=False)
print(f"✅ Duplicates saved to attachment_duplicates.csv ({len(df_duplicates)} files)")

duplicate_files = set(df_duplicates["filename"])
unique_rows = []
for f, fid, ccount, wcount, text in zip(files, feedback_ids, char_counts, word_counts, texts):
    if f not in duplicate_files:
        unique_rows.append({
            "feedbackId": fid,
            "filename": f,
            "clean_filename": clean_filename(f),
            "char_count": ccount,
            "word_count": wcount,
            "attachment_text": clean_attachment_text(text)
        })

df_unique = pd.DataFrame(unique_rows)
df_unique.to_csv("data/attachment_unique.csv", index=False)
print(f"✅ Unique files saved to attachment_unique.csv ({len(df_unique)} files)")
