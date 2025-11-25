#!/usr/bin/env python3
# ============================================
# 📄 EU Feedback Stance Detection Script
# Created by Plop (@plopnl)
# Last updated: 2025-11-21
# Description: Applies zero-shot stance classification to feedback entries
#              using a multilingual transformer model.
# Usage: python 6_stance_detection.py
# ============================================

import os
import pandas as pd
import re
import numpy as np
import evaluate
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import AutoConfig

# =========================
# ⚙️ SETTINGS
# =========================
USE_RULE_BASED = True
USE_ZERO_SHOT = True
USE_SUPERVISED = True
DO_CONFIDENCE_SCORES = True

BATCH_SIZE = 4
NUM_EPOCHS = 6   # adjustable, between 5–8 depending on validation results
LEARNING_RATE = 2e-5
LABEL_SMOOTHING = 0.1   # helps prevent collapse into majority class
USE_CLASS_WEIGHTS = True  # toggle for imbalance correction
ZS_CONFIDENCE_THRESHOLD = 0.60 # Confidence threshold

# Rule-based keyword lists
SUPPORT_KEYWORDS = [
    "i support", "support this", "agree with", "positive change", "good idea",
    "welcome the proposal", "in favor of", "beneficial", "back this",
    "stand behind this", "strongly agree", "positive impact", "worth supporting"
]

OPPOSE_KEYWORDS = [
    "i oppose", "oppose this", "disagree with", "negative impact", "bad idea",
    "against the proposal", "harmful", "not acceptable", "a step backwards",
    "terrible decision", "don’t support this", "disastrous policy",
    "complete failure", "negative consequences", "unintended consequences",
    "increase smoking", "back to smoking", "proportional to risk",
    "proportional to harm", "risk-proportionate", "harm-proportionate"
]

# List of zero-shot models to run
ZERO_SHOT_MODELS = [
    # --- Weak baselines (high "unclear" ~70–78%) ---
#    "facebook/bart-large-mnli",              # weak, but light; ~72% unclear
#    "joeddav/xlm-roberta-large-xnli",        # weak, but light; ~71% unclear
#    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",  # weak; ~78% unclear
#    "cross-encoder/nli-deberta-v3-large",    # weak in your run; ~73% unclear
	
	# --- Moderate candidates (better balance, lower "unclear") ---
#    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",  # moderate; ~51% unclear, strong "against"
#    "vicgalle/xlm-roberta-large-xnli-anli",                  # moderate; ~55% unclear, balanced with ~13% "for"

    # --- Strong performer ---
    "MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33",  # strong; ~16% unclear, ~75% against, ~9% for
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",          # successor to v1.1, expected strong
    "mlburnham/Political_DEBATE_large_v1.0",                # specialized for political debate stance

    # --- Under testing / new suggestions ---
#    "MoritzLaurer/mDeBERTa-v3-large-xnli-multilingual-nli", # larger multilingual variant, promising - not public
#    "microsoft/deberta-v3-xxlarge",                         # very strong if GPU memory allows - not public
]

# Define the model checkpoint you want to fine‑tune (supervised mode)
# 👉 Switch to the strongest zero-shot performer
# MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
MODEL_NAME = "microsoft/deberta-v3-large"
# MODEL_NAME = "xlm-roberta-base"   # or "bert-base-uncased", etc.

# Hypothesis template
HYPOTHESIS_TEMPLATE = "This text expresses a stance that is {} toward the proposal."
# =========================
# ⚙️ PATHS & SETTINGS
# =========================
DATA_DIR = "data"
INPUT_CSV = os.path.join(DATA_DIR, "feedback_details.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "stance_detection")
OUTPUT_CSV = os.path.join(DATA_DIR, "feedback_with_stance.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
#INPUT_CSV = "feedback_details.csv"
#OUTPUT_DIR = "stance_detection"

# =========================
# 📥 LOAD FEEDBACK DATA
# =========================
df = pd.read_csv(INPUT_CSV)

def get_text(row):
    t = str(row.get("feedback_translated", "") or "")
    if t.strip():
        return t.strip()
    return str(row.get("feedback_original", "") or "").strip()

texts = [get_text(row) for _, row in df.iterrows()]

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 📏 RULE-BASED DETECTION
# =========================
def detect_stance_rule(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    if any(phrase in text for phrase in SUPPORT_KEYWORDS):
        return "for"
    if any(phrase in text for phrase in OPPOSE_KEYWORDS):
        return "against"
    return "unclear"

if USE_RULE_BASED:
    df_rule = df.copy()
    df_rule["stance_rule"] = [detect_stance_rule(t) for t in texts]

    # Add mandatory columns with empty defaults so they always exist
    mandatory_cols = [
        "stance_supervised",
        "manual_review",
        "automatic_review",
        "consensus",
        "priority"
    ]
    for col in mandatory_cols:
        if col not in df_rule.columns:
            df_rule[col] = ""

    sr = df_rule["stance_rule"].value_counts().reset_index()
    sr.columns = ["Stance", "Count"]
    sr["Percentage"] = sr["Count"] / sr["Count"].sum() * 100
    print("📊 Rule-based stance summary:")
    print(sr)

    df_rule.to_csv(os.path.join(OUTPUT_DIR, "stance_rule.csv"), index=False)
    print(f"✅ Saved: {OUTPUT_DIR}/stance_rule.csv")

# =========================
# 🤖 ZERO-SHOT DETECTION LOOP
# =========================
if USE_ZERO_SHOT:
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    print(f"⚙️ Using device: {'GPU' if device == 0 else 'CPU'}")

    for model_name in ZERO_SHOT_MODELS:
        print(f"\n🔁 Running zero-shot stance detection with: {model_name}")
        classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device
        )

        candidate_labels = ["for", "against", "unclear"]
        stances = []
        scores_for, scores_against, scores_unclear = [], [], []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            results = classifier(
                batch,
                candidate_labels=candidate_labels,
                hypothesis_template=HYPOTHESIS_TEMPLATE,
                multi_label=False
            )
            for r in results:
                labels = r["labels"]
                scores = r["scores"]

                s_for = scores[labels.index("for")]
                s_against = scores[labels.index("against")]
                s_unclear = scores[labels.index("unclear")]

                top_label = labels[0]
                top_score = scores[0]
                stance = top_label if top_score >= ZS_CONFIDENCE_THRESHOLD else "unclear"
                stances.append(stance)

                if DO_CONFIDENCE_SCORES:
                    scores_for.append(s_for)
                    scores_against.append(s_against)
                    scores_unclear.append(s_unclear)

        df_model = df.copy()
        df_model["stance_zero_shot"] = stances
        df_model["zs_model_name"] = model_name

        if DO_CONFIDENCE_SCORES:
            df_model["zs_for_score"] = scores_for
            df_model["zs_against_score"] = scores_against
            df_model["zs_unclear_score"] = scores_unclear

        sz = pd.Series(stances).value_counts().reset_index()
        sz.columns = ["Stance", "Count"]
        sz["Percentage"] = sz["Count"] / sz["Count"].sum() * 100
        print("📊 Zero-shot stance summary:")
        print(sz)

        model_id = model_name.split("/")[-1].replace("-", "_")
        output_file = os.path.join(OUTPUT_DIR, f"stance_zero_shot_{model_id}.csv")
        df_model.to_csv(output_file, index=False)
        print(f"✅ Saved: {output_file}")

# =========================
# 🧠 SUPERVISED MODEL TRAINING + MANUAL REVIEW LOOP (Binary: for/against)
# =========================
torch.cuda.empty_cache()
SUP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "supervised_model")
os.makedirs(SUP_OUTPUT_DIR, exist_ok=True)

SUP_PATH = os.path.join(OUTPUT_DIR, "stance_supervised.csv")
REVIEW_PATH = os.path.join(SUP_OUTPUT_DIR, "manual_review.csv")

# Precision toggles
USE_BF16 = True  # set False if GPU doesn't support bfloat16 (Ampere+ recommended)
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

if USE_SUPERVISED:
    # Load baseline
    rule_df_full = pd.read_csv(os.path.join(OUTPUT_DIR, "stance_rule.csv")).copy()
    for col in ["manual_review", "automatic_review", "stance_supervised"]:
        if col not in rule_df_full.columns:
            rule_df_full[col] = ""

    if os.path.exists(REVIEW_PATH):
        old_review = pd.read_csv(REVIEW_PATH)
        print("📥 Found existing manual_review.csv, using manual labels where available")
        cols_to_merge = ["feedbackId"]
        if "manual_review" in old_review.columns:
            cols_to_merge.append("manual_review")
        if "automatic_review" in old_review.columns:
            cols_to_merge.append("automatic_review")
        rule_df_full = rule_df_full.merge(
            old_review[cols_to_merge], on="feedbackId", how="left", suffixes=("", "_rev")
        )
        if "manual_review_rev" in rule_df_full.columns:
            rule_df_full["manual_review"] = rule_df_full["manual_review_rev"].fillna(rule_df_full["manual_review"])
            rule_df_full.drop(columns=["manual_review_rev"], inplace=True)
        if "automatic_review_rev" in rule_df_full.columns:
            rule_df_full["automatic_review"] = rule_df_full["automatic_review_rev"].fillna(rule_df_full["automatic_review"])
            rule_df_full.drop(columns=["automatic_review_rev"], inplace=True)
    else:
        print("⚠️ No manual_review.csv found, proceeding without merge")

    LABELS = ["for", "against"]
    stance_rule_filtered = rule_df_full["stance_rule"].apply(lambda x: x if x in LABELS else pd.NA)
    manual_clean = rule_df_full["manual_review"].replace("", pd.NA).apply(lambda x: x if x in LABELS else pd.NA)
    automatic_clean = rule_df_full["automatic_review"].replace("", pd.NA).apply(lambda x: x if x in LABELS else pd.NA)
    rule_df_full["labels_source"] = manual_clean.fillna(automatic_clean).fillna(stance_rule_filtered)

    train_df = rule_df_full[rule_df_full["labels_source"].isin(LABELS)].copy()
    print("📊 Training label distribution:")
    print(train_df["labels_source"].value_counts(normalize=True))
    print("Training subset size:", len(train_df))
    print("Full dataset size:", len(rule_df_full))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        ignore_mismatched_sizes=True,
        torch_dtype=(torch.bfloat16 if USE_BF16 else torch.float32)
    )
    # Fix: disable cache when using gradient checkpointing
    model.config.use_cache = False
    #model.gradient_checkpointing_enable()

    import torch
    class StanceDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels=None):
            self.encodings = encodings
            self.labels = labels
        def __len__(self):
            return len(next(iter(self.encodings.values())))
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            if self.labels is not None:
                item["labels"] = torch.tensor(LABELS.index(self.labels[idx]))
            return item

    import evaluate
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"],
            "f1_per_class": metric_f1.compute(predictions=preds, references=labels, average=None)["f1"]
        }

    labels_for_strat = train_df["labels_source"].tolist()
    try:
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            train_df["feedback_translated"].tolist(),
            labels_for_strat,
            test_size=0.2,
            random_state=42,
            stratify=labels_for_strat
        )
    except ValueError:
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            train_df["feedback_translated"].tolist(),
            labels_for_strat,
            test_size=0.2,
            random_state=42
        )

    MAX_LEN = 256
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LEN)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LEN)
    train_dataset = StanceDataset(train_encodings, train_labels)
    test_dataset = StanceDataset(test_encodings, test_labels)

    training_args = TrainingArguments(
        output_dir=SUP_OUTPUT_DIR,
        logging_dir=os.path.join(SUP_OUTPUT_DIR, "logs"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=2,
        eval_steps=500,
        save_steps=500,
        fp16=False,
        bf16=USE_BF16,
        max_grad_norm=1.0,
        label_smoothing_factor=LABEL_SMOOTHING,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,  # keep as-is per your note
        compute_metrics=compute_metrics,
    )

    if USE_CLASS_WEIGHTS:
        from torch.nn import CrossEntropyLoss
        class_counts = [train_labels.count(lbl) for lbl in LABELS]
        total = sum(class_counts)
        weights = [total / c if c > 0 else 0.0 for c in class_counts]
        weights = torch.tensor(weights, dtype=torch.float).to(model.device)
        loss_fn = CrossEntropyLoss(weight=weights)

        def custom_loss(model, inputs, return_outputs=False, **kwargs):
            labels = inputs["labels"]
            net_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            outputs = model(**net_inputs)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            if return_outputs:
                return (loss, outputs)  # <- return both
            return loss

        trainer.compute_loss = custom_loss

    trainer.train()

    metrics = trainer.evaluate()
    pd.DataFrame([metrics]).to_csv(os.path.join(SUP_OUTPUT_DIR, "stance_supervised_metrics.csv"), index=False)
    print(f"✅ Saved supervised metrics: {SUP_OUTPUT_DIR}/stance_supervised_metrics.csv")

    full_texts = ["" if pd.isna(t) else str(t) for t in rule_df_full["feedback_translated"].tolist()]
    full_encodings = tokenizer(full_texts, truncation=True, padding=True, max_length=MAX_LEN)
    full_dataset = StanceDataset(full_encodings, labels=None)
    preds = trainer.predict(full_dataset)
    import numpy as np
    probs = torch.nn.functional.softmax(torch.tensor(preds.predictions), dim=1).numpy()
    pred_labels = [LABELS[int(np.argmax(p))] if float(np.max(p)) >= 0.6 else "unclear" for p in probs]

    rule_df_full["stance_supervised"] = pred_labels
    print("Training subset size:", len(train_df))
    print("Full dataset size:", len(rule_df_full))
    print("Predictions generated:", len(pred_labels))

    rule_df_full.to_csv(SUP_PATH, index=False)
    print(f"✅ Saved supervised predictions (all rows, all columns preserved): {SUP_PATH}")

    df = pd.read_csv(SUP_PATH)
    print("Saved supervised CSV rows:", len(df))
    print(df["stance_supervised"].value_counts())

# =========================
# 📤 UPDATE MANUAL REVIEW FILE
# =========================
# Always reload rule_df from stance_rule.csv if not already defined
if "rule_df" not in locals():
    rule_df = pd.read_csv(os.path.join(OUTPUT_DIR, "stance_rule.csv"))

# Start with the mandatory columns
review_cols = [
    "feedbackId", "feedback_original", "feedback_translated",
    "stance_rule"
]

# Add stance_supervised if it exists
if "stance_supervised" in rule_df.columns:
    review_cols.append("stance_supervised")

merged = rule_df[review_cols].copy()

# Merge in zero-shot outputs that exist
zs_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("stance_zero_shot")]
for f in zs_files:
    zs_df = pd.read_csv(os.path.join(OUTPUT_DIR, f))
    model_name = f.replace("stance_zero_shot_", "").replace(".csv", "")
    merged[f"stance_zero_shot_{model_name}"] = (
        zs_df.set_index("feedbackId")
        .reindex(merged["feedbackId"])["stance_zero_shot"]
        .values
    )

# Add consensus column (rule vs supervised) safely
if "stance_supervised" in merged.columns:
    merged["consensus"] = [
        r if r == s else "disagreement"
        for r, s in zip(merged["stance_rule"], merged["stance_supervised"])
    ]
else:
    merged["consensus"] = "no_supervised"

# Preserve manual_review edits only for IDs that still exist
if os.path.exists(REVIEW_PATH):
    old_review = pd.read_csv(REVIEW_PATH)
    old_review = old_review[old_review["feedbackId"].isin(merged["feedbackId"])]
    merged = merged.merge(
        old_review[["feedbackId", "manual_review"]],
        on="feedbackId",
        how="left"
    )
else:
    merged["manual_review"] = ""

# Identify all zero-shot stance columns dynamically
zs_cols = [c for c in merged.columns if c.startswith("stance_zero_shot")]

def automatic_review(row):
    preds = [row[c] for c in zs_cols if pd.notna(row.get(c))]
    if len(preds) >= 3 and all(p == preds[0] for p in preds):
        return preds[0]
    return ""

# Add automatic_review column
if len(zs_cols) >= 3:
    merged["automatic_review"] = merged.apply(automatic_review, axis=1)
    print("✅ Automatic review consensus applied (3+ zero-shot models with full agreement).")
else:
    print("⚠️ Skipping automatic review: fewer than 3 zero-shot models available.")

# 🔎 Dynamic priority assignment based on number of "unclear" ratings
stance_cols = [c for c in merged.columns if c.startswith("stance_")]

def assign_priority(row):
    unclear_count = sum(1 for c in stance_cols if row.get(c) == "unclear")
    if unclear_count >= 2:
        return "high"
    elif unclear_count == 1:
        return "medium"
    else:
        return "low"

merged["priority"] = merged.apply(assign_priority, axis=1)

for col in ["manual_review", "automatic_review", "stance_supervised", "consensus", "priority"]:
    if col not in merged.columns:
        merged[col] = ""

# Save updated manual review file
merged.to_csv(REVIEW_PATH, index=False)
print(f"✅ Cleaned and updated manual review file: {REVIEW_PATH}")

# Also update stance_rule.csv so supervised always sees the full schema
merged.to_csv(os.path.join(OUTPUT_DIR, "stance_rule.csv"), index=False)
print(f"✅ Synced stance_rule.csv with mandatory columns")

