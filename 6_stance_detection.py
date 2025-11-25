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
USE_ZERO_SHOT = False
USE_SUPERVISED = True
DO_CONFIDENCE_SCORES = True
OVERSAMPLE_SUPERVISED = True
LOG_FLIPS_TO_CSV = True

BATCH_SIZE = 4
NUM_EPOCHS = 6   # adjustable, between 5–8 depending on validation results
LEARNING_RATE = 2e-5
LABEL_SMOOTHING = 0.1   # helps prevent collapse into majority class
USE_CLASS_WEIGHTS = True  # toggle for imbalance correction
ZS_CONFIDENCE_THRESHOLD = 0.60 # Confidence threshold

SUPPORT_KEYWORDS = [
    # Direct endorsements
    "i support", "support this", "agree with", "positive change", "good idea",
    "welcome the proposal", "welcome the opportunity", "in favor of", "beneficial",
    "back this", "stand behind this", "strongly agree", "positive impact",
    "worth supporting", "welcome directive", "agree with commission", "positive step",
    "aligned with cancer strategy", "necessary harmonisation", "EU should lead",
    "stronger regulation is good", "protect youth", "reduce smoking through taxation",
    "public health priority",
    # NGO/academic supportive framing
    "beating cancer plan", "smoke-free target", "smoke-free generation",
    "unique opportunity to improve public health", "aligned with WHO recommendations",
    "harmonisation of excise duty rates", "coherent excise system",
    "regular increases in tobacco taxes", "adopt the proposal without delay",
    "ambitious harmonised excise system", "weighted average price",
    "future-proof excise system"
]

OPPOSE_KEYWORDS = [
    # Direct opposition
    "i oppose", "oppose this", "disagree with", "negative impact", "bad idea",
    "against the proposal", "harmful", "not acceptable", "a step backwards",
    "terrible decision", "don’t support this", "disastrous policy", "complete failure",
    "negative consequences", "unintended consequences", "should not", "reject",
    # Critique patterns
    "risks achieving the opposite", "discourage switching", "equal taxation",
    "ignores reality", "remove the financial incentive", "undermine harm reduction",
    "inappropriate", "disproportionate", "increase illicit trade", "increase smoking",
    "black market", "parallel market", "illicit trade", "contraband", "smuggling",
    # Sweden / harm-reduction invoked against EU proposal
    "sweden shows", "sweden has the lowest proportion of smokers", "lower healthcare costs",
    "harm reduction works", "snus helped me stop smoking", "white snus",
    "nicotine portions/pouches", "snus helped me quit", "evidence from sweden",
    # Policy consequences
    "no age limits", "consumer safety falls", "punish consumers", "punish a tool",
    "symbolic policy", "symbolic politics", "unreasonable tax", "contradictory policy",
    "absurd proposal", "absurd", "stop punishing", "freedom of choice",
    "snus is banned in the eu", "taxing while banned", "facts and fairness",
    "proportionality", "risk-based taxation", "bureaucracy", "expensive mistake",
    "kills freedom", "stop hunting snus", "against harmonisation",
    # Academic harm-reduction critiques
    "risk-based regulation", "non-combustible alternatives", "smokeless products are less harmful",
    "discourages switching", "penalising innovation", "proportional taxation",
    "counterproductive effects", "regressive impact", "delegated acts"
]

HARM_REDUCTION_SIGNALS = [
    "risk-proportionate", "proportional to risk", "harm-proportionate",
    "minimal excise duty on vaping", "minimal excise duty on pouches",
    "low excise on heated tobacco", "incentive to switch",
    "lowest proportion of smokers", "reduce healthcare costs",
    "freedom of choice", "risk-based taxation", "white snus / nicotine pouches",
    "harm reduction works", "safer alternatives", "evidence from sweden",
    "scientific evidence shows", "switching to less harmful alternatives",
    "heated tobacco", "e-cigarettes", "nicotine pouches", "absence of combustion",
    "reduce exposure to harmful toxins", "innovation in non-combustible products"
]



# List of zero-shot models to run
ZERO_SHOT_MODELS = [
    # --- Strong performer ---
    "MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33",  # strong; ~16% unclear, ~75% against, ~9% for
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",          # successor to v1.1, expected strong
    "mlburnham/Political_DEBATE_large_v1.0",                # specialized for political debate stance
]

# Define the model checkpoint you want to fine‑tune (supervised mode)
# 👉 Switch to the strongest zero-shot performer
# MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
MODEL_NAME = "microsoft/deberta-v3-large"
# MODEL_NAME = "xlm-roberta-base"   # or "bert-base-uncased", etc.

# Hypothesis template (keep proposal focus explicit)
HYPOTHESIS_TEMPLATE = "This text expresses a stance that is {} toward the Commission’s proposal as written."

# =========================
# ⚙️ PATHS & SETTINGS
# =========================
DATA_DIR = "data"
INPUT_CSV = os.path.join(DATA_DIR, "feedback_details.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "stance_detection")
OUTPUT_CSV = os.path.join(DATA_DIR, "feedback_with_stance.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 📥 LOAD FEEDBACK DATA
# =========================
df = pd.read_csv(INPUT_CSV)

# --- Merge attachments if available ---
try:
    df_unique = pd.read_csv("data/attachment_unique.csv")
    df_dupes = pd.read_csv("data/attachment_duplicates.csv")
    df_attach = pd.concat([df_unique, df_dupes], ignore_index=True)

    # Group attachment texts per feedbackId
    df_attach_text = (
        df_attach.groupby("feedbackId")["attachment_text"]
        .apply(lambda x: "\n\n".join(x.dropna().astype(str)))
        .reset_index()
    )

    # Merge into base
    df = pd.merge(df, df_attach_text, on="feedbackId", how="left")

    # Build unified field: feedback + attachments
    df["feedback_fulltext"] = (
        df["feedback_translated"].fillna(df.get("feedback_original", "")).astype(str)
        + "\n\n"
        + df["attachment_text"].fillna("").astype(str)
    )

    merged_count = df_attach_text["feedbackId"].nunique()
    print(f"✅ Attachment merge successful: merged {merged_count} feedbackIds with attachments")

except Exception as e:
    print(f"⚠️ Attachment merge skipped: {e}")
    df["feedback_fulltext"] = df["feedback_translated"].fillna(df.get("feedback_original", "")).astype(str)


# --- Text getter prefers fulltext ---
def get_text(row):
    t = str(row.get("feedback_fulltext", "") or "").strip()
    if t:
        return t
    t = str(row.get("feedback_translated", "") or "").strip()
    if t:
        return t
    return str(row.get("feedback_original", "") or "").strip()

texts = [get_text(row) for _, row in df.iterrows()]

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 🔎 Dual-signal + refinement utilities
# =========================
def extract_sections(text: str, head_chars=800, tail_chars=1200):
    t = (text or "").strip()
    if not t:
        return "", "", ""
    if len(t) <= head_chars + tail_chars:
        return t[:head_chars], t, t[-tail_chars:]
    head = t[:head_chars]
    tail = t[-tail_chars:]
    body = t[head_chars:-tail_chars] if len(t) > (head_chars + tail_chars) else t
    return head, body, tail

def signal_score(text: str, phrases: list):
    tl = text.lower()
    return sum(p in tl for p in phrases)

def classify_goal_instrument(text: str):
    head, body, tail = extract_sections(text)
    blocks = [head, body, tail]
    goal_support = max(signal_score(b, SUPPORT_KEYWORDS) for b in blocks)
    instrument_oppose = max(
        signal_score(b, OPPOSE_KEYWORDS) + signal_score(b, HARM_REDUCTION_SIGNALS)
        for b in blocks
    )
    goal_stance = "For" if goal_support > 0 else "Unclear"
    proposal_stance = "Against" if instrument_oppose > 0 else "Unclear"
    return goal_stance, proposal_stance

def refine_proposal_stance(text: str, raw_stance: str):
    # Minimum length guard: very short texts → Unclear
    if len(text.split()) < 5 or len(text) < 30:
        return "unclear"
        
    # Prioritize critique in body/tail; override For → Against if strong instrument opposition found
    head, body, tail = extract_sections(text)
    oppose_score = (
        signal_score(body, OPPOSE_KEYWORDS) + signal_score(tail, OPPOSE_KEYWORDS) +
        signal_score(body, HARM_REDUCTION_SIGNALS) + signal_score(tail, HARM_REDUCTION_SIGNALS)
    )
    # Also guard against negations like "we support the goal, but..."
    has_contrast = any(k in text.lower() for k in ["but", "however", "nevertheless", "yet", "nonetheless"])
    if raw_stance.lower() == "for" and (oppose_score > 0 or has_contrast):
        return "against"
    return raw_stance
    
# =========================
# 🚫 Negation override
# =========================
NEGATION_PATTERNS = [
    "i do not agree",
    "do not agree",
    "i disagree",
    "i oppose",
    "do not support",
    "don’t support",
    "not acceptable",
    "strong opposition"
]

def negation_override(text: str) -> str | None:
    """
    Returns 'against' if a clear negation phrase is found,
    otherwise None (let the model decide).
    """
    t = text.lower()
    for pat in NEGATION_PATTERNS:
        if pat in t:
            return "against"
    return None

# =========================
# 📏 RULE-BASED DETECTION
# =========================
def detect_stance_rule(text):
    # Negation override first
    neg = negation_override(text)
    if neg:
        return neg
        
    text_norm = re.sub(r"[^\w\s]", " ", str(text).lower())
    # First, dual-signal classification
    goal, proposal = classify_goal_instrument(text_norm)
    if proposal == "Against":
        return "against"
    if goal == "For":
        return "for"
    # Fallback to simple lexicon if dual-signal was inconclusive
    if any(phrase in text_norm for phrase in SUPPORT_KEYWORDS):
        return "for"
    if any(phrase in text_norm for phrase in OPPOSE_KEYWORDS) or any(phrase in text_norm for phrase in HARM_REDUCTION_SIGNALS):
        return "against"
    return "unclear"

if USE_RULE_BASED:
    df_rule = df.copy()
    df_rule["stance_rule"] = [detect_stance_rule(t) for t in texts]

    # Dual labels for auditability (optional, not used downstream unless desired)
    goal_instrument = [classify_goal_instrument(t) for t in texts]
    df_rule["goal_stance"], df_rule["proposal_stance"] = zip(*goal_instrument)

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
            for text, r in zip(batch, results):
                labels = r["labels"]
                scores = r["scores"]

                s_for = scores[labels.index("for")]
                s_against = scores[labels.index("against")]
                s_unclear = scores[labels.index("unclear")]

                top_label = labels[0]
                top_score = scores[0]
                stance_raw = top_label if top_score >= ZS_CONFIDENCE_THRESHOLD else "unclear"

                # Root fix: refine proposal stance using contrast + instrument signals
                stance_refined = refine_proposal_stance(text, stance_raw)

                stances.append(stance_refined)
                if DO_CONFIDENCE_SCORES:
                    scores_for.append(s_for)
                    scores_against.append(s_against)
                    scores_unclear.append(s_unclear)

        df_model = df.copy()
        df_model["stance_zero_shot"] = stances
        df_model["zs_model_name"] = model_name

        # Dual labels for auditability (optional)
        goal_instrument = [classify_goal_instrument(t) for t in texts]
        df_model["goal_stance"], df_model["proposal_stance"] = zip(*goal_instrument)

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
    # Load baseline labels (rule-based output)
    rule_df_full = pd.read_csv(os.path.join(OUTPUT_DIR, "stance_rule.csv")).copy()
    for col in ["manual_review", "automatic_review", "stance_supervised"]:
        if col not in rule_df_full.columns:
            rule_df_full[col] = ""

    # Merge any existing manual review labels
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

    # Prepare training labels (binary: for/against)
    LABELS = ["for", "against"]
    stance_rule_filtered = rule_df_full["stance_rule"].apply(lambda x: x if x in LABELS else pd.NA)
    manual_clean = rule_df_full["manual_review"].replace("", pd.NA).apply(lambda x: x if x in LABELS else pd.NA)
    automatic_clean = rule_df_full["automatic_review"].replace("", pd.NA).apply(lambda x: x if x in LABELS else pd.NA)

    # 1) Base label source priority: manual > automatic > rule
    rule_df_full["labels_source_raw"] = manual_clean.fillna(automatic_clean).fillna(stance_rule_filtered)

    # 2) Proposal-focused override for training labels (root fix)
    #    If a text contains instrument opposition (in body/tail), flip raw "for" → "against"
    def proposal_override(text, label):
        if pd.isna(label):
            return pd.NA
        # Minimum length guard
        if len(str(text).split()) < 5 or len(str(text)) < 30:
            return "unclear"
        return refine_proposal_stance(str(text or ""), str(label)).lower()

    # Prefer translated text; fallback to original
    text_series = rule_df_full["feedback_translated"].fillna(rule_df_full.get("feedback_original", ""))

    rule_df_full["labels_source"] = [
        proposal_override(t, lbl).lower() if (not pd.isna(lbl) and lbl in LABELS) else pd.NA
        for t, lbl in zip(text_series, rule_df_full["labels_source_raw"])
    ]


    train_df = rule_df_full[rule_df_full["labels_source"].isin(LABELS)].copy()
    print("📊 Training label distribution (after proposal override):")
    print(train_df["labels_source"].value_counts(normalize=True))
    print("Training subset size:", len(train_df))
    print("Full dataset size:", len(rule_df_full))

    # --- Optional oversampling ---
    if OVERSAMPLE_SUPERVISED:
        from sklearn.utils import resample

        df_majority = train_df[train_df["labels_source"] == "against"]
        df_minority = train_df[train_df["labels_source"] == "for"]

        df_minority_oversampled = resample(
            df_minority,
            replace=True,
            n_samples=len(df_majority),
            random_state=42
        )

        train_df = pd.concat([df_majority, df_minority_oversampled])
        print("📊 Balanced training label distribution (oversampled):")
        print(train_df["labels_source"].value_counts(normalize=True))
        print("Balanced training subset size:", len(train_df))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        ignore_mismatched_sizes=True,
        torch_dtype=(torch.bfloat16 if USE_BF16 else torch.float32)
    )
    model.config.use_cache = False
    # model.gradient_checkpointing_enable()  # optional if memory-bound

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
            train_df["feedback_translated"].fillna(train_df.get("feedback_original", "")).tolist(),
            labels_for_strat,
            test_size=0.2,
            random_state=42,
            stratify=labels_for_strat
        )
    except ValueError:
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            train_df["feedback_translated"].fillna(train_df.get("feedback_original", "")).tolist(),
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
                return (loss, outputs)
            return loss

        trainer.compute_loss = custom_loss

    trainer.train()

    metrics = trainer.evaluate()
    pd.DataFrame([metrics]).to_csv(os.path.join(SUP_OUTPUT_DIR, "stance_supervised_metrics.csv"), index=False)
    print(f"✅ Saved supervised metrics: {SUP_OUTPUT_DIR}/stance_supervised_metrics.csv")

    # Predict on full dataset (prefer translated text, fallback to original)
    full_texts = rule_df_full["feedback_translated"].fillna(rule_df_full.get("feedback_original", "")).astype(str).tolist()
    full_encodings = tokenizer(full_texts, truncation=True, padding=True, max_length=MAX_LEN)
    full_dataset = StanceDataset(full_encodings, labels=None)
    preds = trainer.predict(full_dataset)

    probs = torch.nn.functional.softmax(torch.tensor(preds.predictions), dim=1).numpy()
    pred_labels_raw = [LABELS[int(np.argmax(p))] if float(np.max(p)) >= 0.6 else "unclear" for p in probs]
    # Build text field for flip routines: prefer translated, else original
    texts_for_flips = (
        rule_df_full["feedback_translated"].fillna("").replace("nan", "").astype(str)
    )
    texts_for_flips = texts_for_flips.where(
        texts_for_flips.str.strip() != "",
        rule_df_full["feedback_original"].fillna("").astype(str)
    )

    # --- Diagnostic counters for keyword matches ---
    support_hits = sum(any(k in t.lower() for k in SUPPORT_KEYWORDS) for t in texts_for_flips)
    oppose_hits = sum(any(k in t.lower() for k in OPPOSE_KEYWORDS) for t in texts_for_flips)
    harm_hits    = sum(any(k in t.lower() for k in HARM_REDUCTION_SIGNALS) for t in texts_for_flips)

    print("📊 Keyword match diagnostics:")
    print(f"  SUPPORT_KEYWORDS matched: {support_hits}")
    print(f"  OPPOSE_KEYWORDS matched:  {oppose_hits}")
    print(f"  HARM_REDUCTION_SIGNALS:   {harm_hits}")


    # Debug: show how many rows used translated vs original
    num_translated = (rule_df_full["feedback_translated"].fillna("").str.strip() != "").sum()
    num_original_fallback = len(rule_df_full) - num_translated
    print(f"🔎 Flip input source: {num_translated} translated, {num_original_fallback} original fallbacks")

    # Apply negation override before refinement
    pred_labels_with_neg = []
    neg_flips = []
    for fid, text, lbl in zip(rule_df_full["feedbackId"], texts_for_flips, pred_labels_raw):
        neg = negation_override(text)
        if neg and lbl != "against":
            pred_labels_with_neg.append(neg)
            neg_flips.append({"feedbackId": fid, "text": text, "raw": lbl, "flipped_to": neg})
        else:
            pred_labels_with_neg.append(lbl)

    print(f"🔎 Negation overrides applied: {len(neg_flips)} of {len(pred_labels_raw)} predictions")

    # Root fix: refine model predictions with proposal-focused override
    pred_labels = []
    proposal_flips = []
    for fid, text, lbl in zip(rule_df_full["feedbackId"], texts_for_flips, pred_labels_with_neg):
        refined = refine_proposal_stance(text, lbl)
        pred_labels.append(refined)
        if refined != lbl:
            proposal_flips.append({"feedbackId": fid, "text": text, "raw": lbl, "flipped_to": refined})

    print(f"🔎 Proposal-focused refinement flips: {len(proposal_flips)} of {len(pred_labels)} predictions")

    if LOG_FLIPS_TO_CSV:
        flips_df = pd.DataFrame(neg_flips + proposal_flips)
        flips_path = os.path.join(SUP_OUTPUT_DIR, "stance_flips.csv")
        flips_df.to_csv(flips_path, index=False)
        print(f"✅ Saved flip log: {flips_path}")

    # Audit columns: show flips and dual-signal labels
    rule_df_full["stance_supervised_raw"] = pred_labels_raw
    rule_df_full["stance_supervised"] = pred_labels

    dual_labels = [classify_goal_instrument(t) for t in texts_for_flips]
    rule_df_full["goal_stance"], rule_df_full["proposal_stance"] = zip(*dual_labels)

    flips = sum(1 for a, b in zip(pred_labels_raw, pred_labels) if a != b)
    print(f"🔎 Proposal-focused refinement flips: {flips} of {len(pred_labels)} predictions")

    print("Training subset size:", len(train_df))
    print("Full dataset size:", len(rule_df_full))
    print("Predictions generated:", len(pred_labels))

    rule_df_full.to_csv(SUP_PATH, index=False)
    print(f"✅ Saved supervised predictions (all rows, all columns preserved): {SUP_PATH}")

    df = pd.read_csv(SUP_PATH)
    print("Saved supervised CSV rows:", len(df))
    print(df["stance_supervised"].value_counts())

    # print summary
    summary = df["stance_supervised"].value_counts(normalize=True) * 100
    print(summary)


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

