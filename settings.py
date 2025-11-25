"""
Settings for the EU consultation feedback analysis pipeline.

Fill in the PUBLICATION SETTINGS block below for your consultation, then run:
    ./run_pipeline.sh

All scripts import these with: from settings import *
"""

import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))

# =========================
# 🌐 PUBLICATION SETTINGS   ← edit these for your consultation
# =========================
DEFAULT_PUBLICATION_ID = 20315
INITIATIVE_ID          = 12645
INITIATIVE_SLUG        = "12645-Tobacco-taxation-excise-duties-for-manufactured-tobacco-products-updated-rules-_en"
INITIATIVE_URL         = (
    "https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives/"
    + INITIATIVE_SLUG
)
CONSULTATION_START = "03 September 2025"
CONSULTATION_END   = "31 October 2025"

INITIATIVE_URL_BASE = "https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives/"

# =========================
# 📁 PATHS
# =========================
DATA_DIR       = _os.path.join(_ROOT, "data")
FIGURES_DIR    = _os.path.join(_ROOT, "figures")
ATTACHMENT_DIR = _os.path.join(_ROOT, "attachments")

FEEDBACK_DETAILS_FILE   = _os.path.join(DATA_DIR, "feedback_details.csv")
FEEDBACK_WITH_LENGTHS   = _os.path.join(DATA_DIR, "feedback_details_with_lengths.csv")
FEEDBACK_WITH_STANCE    = _os.path.join(DATA_DIR, "feedback_with_stance.csv")
FEEDBACK_IDS_FILE       = _os.path.join(DATA_DIR, "feedback_ids.csv")
ATTACHMENT_UNIQUE_FILE  = _os.path.join(DATA_DIR, "attachment_unique.csv")
ATTACHMENT_DUP_FILE     = _os.path.join(DATA_DIR, "attachment_duplicates.csv")
STANCE_DIR              = _os.path.join(DATA_DIR, "stance_detection")
STANCE_SUPERVISED_FILE  = _os.path.join(DATA_DIR, "stance_detection", "stance_supervised.csv")
STANCE_RULE_FILE        = _os.path.join(DATA_DIR, "stance_detection", "stance_rule.csv")
BOT_DETECTION_FILE      = _os.path.join(DATA_DIR, "bot_detection_v4.csv")

FEEDBACK_SIMILARITY_TOPN    = _os.path.join(DATA_DIR, "feedback_similarity_topn.csv")
ATTACHMENT_SIMILARITY_TOPN  = _os.path.join(DATA_DIR, "attachment_similarity_topn.csv")
ATTACHMENT_COMBINED_TOPN    = _os.path.join(DATA_DIR, "attachment_combined_topn.csv")
SIMILARITY_WITH_ATTACHMENTS = _os.path.join(DATA_DIR, "similarity_with_attachments.csv")
KEYWORD_COUNTS_FILE         = _os.path.join(DATA_DIR, "keyword_counts.csv")
BIGRAM_COUNTS_FILE          = _os.path.join(DATA_DIR, "bigram_counts_translated.csv")
FEEDBACK_DUPLICATES_FILE    = _os.path.join(DATA_DIR, "feedback_duplicates.csv")
BINNED_TEXT_DUP_FILE        = _os.path.join(DATA_DIR, "binned_feedback_duplicates.csv")
BINNED_ATTACH_DUP_FILE      = _os.path.join(DATA_DIR, "binned_attachment_duplicates.csv")
SPLIT_BY_USERTYPE_DIR       = _os.path.join(DATA_DIR, "split_by_usertype")
FEEDBACK_BINNED_FILE        = _os.path.join(DATA_DIR, "feedback_binned.csv")
PER_COUNTRY_DIR             = _os.path.join(DATA_DIR, "feedback_by_country")
WORD_FREQ_BY_COUNTRY_DIR    = _os.path.join(DATA_DIR, "word_frequencies_by_country")
WORD_FREQ_BY_LANGUAGE_DIR   = _os.path.join(DATA_DIR, "word_frequencies_by_language")

SUP_OUTPUT_DIR    = _os.path.join(DATA_DIR, "stance_detection", "supervised_model")
REVIEW_PATH       = _os.path.join(DATA_DIR, "stance_detection", "supervised_model", "manual_review.csv")
PSEUDO_LABEL_FILE = _os.path.join(DATA_DIR, "stance_detection", "supervised_model", "pseudo_labels.csv")
RUN_METRICS_FILE  = _os.path.join(DATA_DIR, "stance_detection", "supervised_model", "run_metrics.csv")
CLAUDE_LABELS_FILE = _os.path.join(DATA_DIR, "stance_detection", "claude_labels.csv")

RULES_BASE_FILE      = _os.path.join(_ROOT, "rules_base.json")
EMERGENCY_QUEUE_FILE = _os.path.join(DATA_DIR, "stance_detection", "emergency_queue.json")
TEST_RUN_DIR         = _os.path.join(DATA_DIR, "stance_detection", "test_run")

# =========================
# 📊 PLOT FONT SIZES
# =========================
TITLE_FONTSIZE      = 14
LABEL_FONTSIZE      = 9
LEGEND_FONTSIZE     = 7
FOOTER_FONTSIZE     = 7
TICK_FONTSIZE       = 9
ANNOTATION_FONTSIZE = 9

FIGURE_DPI  = 200
SAVEFIG_DPI = 300

# =========================
# 📝 FOOTER
# =========================
FOOTER_CREATED_BY = (
    "EU Have Your Say Scraper & Analysis Toolkit"
    "  ·  https://github.com/plopnl/EU-Have-your-say-webscraper-and-analysis-toolkit"
)

FOOTER_LINE_1 = f"Data source: Official consultation contributions ({CONSULTATION_START} – {CONSULTATION_END})"
FOOTER_LINE_2 = f"URL: {INITIATIVE_URL}"
FOOTER_LINES  = [FOOTER_LINE_1, FOOTER_LINE_2]

FOOTER_LINES_PLOT = [
    FOOTER_LINE_1,
    "URL: " + INITIATIVE_URL_BASE,
    "     " + INITIATIVE_SLUG,
    FOOTER_CREATED_BY,
]
FOOTER_PLOT_TEXT = "\n".join(FOOTER_LINES_PLOT)


def make_footer_text(extra_lines=None, fig_width_inches=10, fontsize=7):
    import sys
    frame = sys._getframe(1)
    globs = frame.f_globals

    footer_line_1   = globs.get("FOOTER_LINE_1", "")
    initiative_url  = globs.get("INITIATIVE_URL", "")
    initiative_slug = globs.get("INITIATIVE_SLUG", "")

    char_width_in = fontsize * 0.6 / 72.0
    max_chars = max(60, int(fig_width_inches / char_width_in))

    lines = list(extra_lines) if extra_lines else []
    lines.append(footer_line_1)
    url_full = "URL: " + initiative_url
    if len(url_full) <= max_chars:
        lines.append(url_full)
    else:
        lines.append("URL: " + INITIATIVE_URL_BASE)
        lines.append("     " + initiative_slug)
    lines.append(FOOTER_CREATED_BY)
    return "\n".join(lines)


# =========================
# 🎨 COLOUR SCHEMES
# =========================
STANCE_COLORS = {
    "against": "#d62728",
    "for":     "#2ca02c",
    "unclear": "#9e9e9e",
}

BAR_COLORS = {
    "for":     "#66c2a5",
    "against": "#fc8d62",
    "unclear": "#8da0cb",
}

LINE_COLORS = {
    "for":     "#1b9e77",
    "against": "#d95f02",
    "unclear": "#7570b3",
}

# =========================
# 🔍 ANALYSIS SETTINGS
# =========================
DUPLICATE_THRESHOLD = 0.95
SHORT_WORD_LIMIT    = 10
BURST_GAP_SECONDS   = 300

TOP_N                = 5
SIMILARITY_THRESHOLD = 0.0

SHOW_STARS            = False
SUBMISSION_LIMIT_DAYS = 59
BOT_WINDOW_SEC        = 300

SMOOTHING_SIGMA       = 1.5
SIMILARITY_THRESHOLDS = [0.85, 0.95]
BIN_COUNT             = 50

SEPARATE_BY_COUNTRY  = True
SEPARATE_BY_LANGUAGE = True

CUTOFF_ATTACHMENT = 0.95
CUTOFF_TEXT       = 0.95

TOP_X_KEYWORDS      = 20
TOP_X_COUNTRIES     = 10
TOP_GROUPS          = 10
ZOOM_MARGIN         = 0.02
LINE_SPACING_FOOTER = 1.5

BIN_LABELS = ["Ultra-short (0–10)", "Short (11–50)", "Medium (51–200)", "Long (>200)"]
BIN_EDGES  = [0, 10, 50, 200, float("inf")]
BIN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
STANCE_ORDER = ["Against", "For", "Unclear"]

# =========================
# 🌐 STAGE 1 — API FETCH
# =========================
LANGUAGE              = "EN"
PAGE_SIZE             = 10
PAGE_SIZE_THREADED    = 50
FETCH_THREADS         = 20
SLEEP_SECONDS         = 0.5
SLEEP_SECONDS_DETAILS = 0.2

# =========================
# ⚗️  TEST RUN SETTINGS
# =========================
TEST_RUN     = False
TEST_RUN_DIR = _os.path.join(DATA_DIR, "stance_detection", "test_run")

# =========================
# 🤖 STAGE 3 — STANCE DETECTION
# =========================
USE_RULE_BASED        = True
USE_ZERO_SHOT         = False
USE_CLAUDE_LABELS     = True
USE_SUPERVISED        = True
DO_CONFIDENCE_SCORES  = True
OVERSAMPLE_SUPERVISED = True
USE_CLASS_WEIGHTS     = True
USE_SAMPLE_WEIGHTS    = True
FORCE_RETRAIN         = False
MIN_TRAIN_SAMPLES     = 2000

MODEL_BACKUP_COPIES    = 3
PIPELINE_BACKUP_COPIES = 5

GUARD_WARN_THRESHOLD     = 25.0
GUARD_COLLAPSE_THRESHOLD = 50.0

BATCH_SIZE      = 4
NUM_EPOCHS      = 6
LEARNING_RATE   = 2e-5
LABEL_SMOOTHING = 0.1
USE_BF16        = True
MAX_LEN         = 256

ZS_CONFIDENCE_THRESHOLD = 0.60
PSEUDO_ZS_MIN_MODELS    = 3
PSEUDO_CONF_THRESHOLD   = 0.85

MODEL_NAME = "microsoft/deberta-v3-large"
ZERO_SHOT_MODELS = [
    "MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33",
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    "mlburnham/Political_DEBATE_large_v1.0",
]
HYPOTHESIS_TEMPLATE = "This text expresses a stance that is {} toward the proposal."
