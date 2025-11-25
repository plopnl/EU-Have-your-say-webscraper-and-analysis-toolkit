#!/bin/bash
set -e  # stop on error

# Activate venv
VENV="$HOME/venvs/eu_scraper"
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
else
    echo "ERROR: venv not found at $VENV" >&2
    exit 1
fi

# 🎨 Colors
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
RESET="\033[0m"

echo -e "${GREEN}🚀 EU Feedback Analysis Pipeline Menu${RESET}"

# Helper to run a step
run_step() {
    local label="$1"
    shift
    echo -e "${BLUE}▶ Running: $label${RESET}"
    "$@"
    echo -e "${GREEN}✔ Completed: $label${RESET}\n"
}

# Step 0: Resolve IDs from active_consultation.txt
ACTIVE_FILE="active_consultation.txt"
if [ -f "$ACTIVE_FILE" ]; then
    ACTIVE_DIR=$(cat "$ACTIVE_FILE" | tr -d '[:space:]')
else
    ACTIVE_DIR="consultations/12645/20315"
fi
# Path is consultations/<initiative_id>/<publication_id>
DEFAULT_INITIATIVE_ID=$(echo "$ACTIVE_DIR" | cut -d'/' -f2)
DEFAULT_PUBLICATION_ID=$(echo "$ACTIVE_DIR" | cut -d'/' -f3)

echo -e "${YELLOW}=== 0️⃣ Setup IDs ===${RESET}"
echo -e "Active consultation: ${BLUE}$ACTIVE_DIR${RESET}"
read -p "Enter PUBLICATION_ID (default $DEFAULT_PUBLICATION_ID): " PUBLICATION_ID
PUBLICATION_ID=${PUBLICATION_ID:-$DEFAULT_PUBLICATION_ID}

read -p "Enter INITIATIVE_ID (default $DEFAULT_INITIATIVE_ID): " INITIATIVE_ID
INITIATIVE_ID=${INITIATIVE_ID:-$DEFAULT_INITIATIVE_ID}

echo -e "${GREEN}✔ Using PUBLICATION_ID=$PUBLICATION_ID INITIATIVE_ID=$INITIATIVE_ID${RESET}\n"

while true; do
    echo -e "${YELLOW}=== Choose a pipeline stage ===${RESET}"
    echo "1) Fetch & Collect Raw Data"
    echo "2) Deduplication & Cleaning"
    echo "3) Stance Detection"
    echo "4) Similarity Analysis"
    echo "5) Bot Detection"
    echo "6) Content Analysis"
    echo "7) Stance Visualizations"
    echo "8) Final Visualizations"
    echo "9) Run ALL stages (safe mode)"
    echo ""
    echo "a) Rerun stance detection + metrics"
    echo "p) Regenerate plots (stages 5 + 6 + 7 + 8)"
    echo "n) Nightly run: stance → plots (5 + 6 + 7 + 8)"
    echo ""
    echo "t) Test stance run (stats only — no production files modified)"
    echo ""
    echo "d) Load SQLite database (load_db.py)"
    echo ""
    echo "b) Backup"
    echo ""
    echo "0) Exit"
    read -p "Enter choice [0-9/a/d/p/n/b/t]: " choice

    case $choice in
        1)
            echo -e "${YELLOW}=== 📥 Step 1: Fetch & Collect Raw Data ===${RESET}"
            run_step "Collect feedback IDs" \
                python 1_get_feedback_ids.py --publication-id "$PUBLICATION_ID"

            run_step "Fetch feedback details (retry)" \
                python 1_fetch_feedback_details_with_retry.py --initiative-id "$INITIATIVE_ID"

            run_step "Download attachments" \
                python 1_download_attachments.py

            run_step "Check missing IDs" \
                python 1_look_missing_ids.py
            ;;
        2)
            echo -e "${YELLOW}=== 🧹 Step 2: Deduplication & Cleaning ===${RESET}"
            run_step "Count characters" python 2_count_characters.py
            run_step "Feedback similarity matrix" python 2_feedback_similarity_matrix.py
            run_step "Split duplicate vs unique" python 2_split_duplicate_unique.py
            run_step "Compare attachment texts" python 2_compare_attachment_texts.py
            run_step "Attachment similarity matrix" python 2_attachments_similarity_matrix.py
            run_step "Translate non-English attachments" python 2_translate_attachments.py
            ;;
        3)
            echo -e "${YELLOW}=== 🧭 Step 3: Stance Detection ===${RESET}"
            run_step "Stance detection" python 3_stance_detection.py
            run_step "Log metrics" python 3_log_metrics.py

            read -p "⚠️  Run 3_manage_zero_shot.py? (y/N): " confirm_zero
            if [[ "$confirm_zero" == "y" || "$confirm_zero" == "Y" ]]; then
                run_step "Manage zero-shot runs" python 3_manage_zero_shot.py
            else
                echo "⏩ Skipped 3_manage_zero_shot.py"
            fi

            read -p "⚠️  Run 3_reset_supervised.py (this will wipe supervised stance)? (y/N): " confirm_reset
            if [[ "$confirm_reset" == "y" || "$confirm_reset" == "Y" ]]; then
                run_step "Reset supervised pipeline" python 3_reset_supervised.py
            else
                echo "⏩ Skipped 3_reset_supervised.py"
            fi
            ;;
        4)
            echo -e "${YELLOW}=== 🔍 Step 4: Similarity Analysis ===${RESET}"
            run_step "Feedback similarity analysis" python 4_similarity_analysis.py
            run_step "Attachment similarity analysis" python 4_similarity_analysis_attachments.py
            ;;
        5)
            echo -e "${YELLOW}=== 🤖 Step 5: Bot Detection ===${RESET}"
            run_step "Bot detection" python 5_bot_detection_v4.py
            run_step "Bot detection analysis" python 5_bot_detection_v4_analysis.py
            run_step "Bot detection analysis v4.1" python 5_bot_detection_v4.1_analysis.py
            ;;
        6)
            echo -e "${YELLOW}=== 📊 Step 6: Content Analysis ===${RESET}"
            run_step "Submission volume by day" python 6_submission_volume_by_day.py
            run_step "Word frequency counter" python 6_word_frequency_counter.py
            run_step "Bigram analysis" python 6_bigrams.py
            run_step "Country summary" python 6_country.py
            run_step "Keyword counter" python 6_keyword_counter.py
            ;;
        7)
            echo -e "${YELLOW}=== 📈 Step 7: Stance Visualizations ===${RESET}"
            run_step "Stance analysis" python 7_stance_analysis.py
            run_step "Stance by usertype" python 7_stance_by_usertype.py
            run_step "Stance by usertype (dedup weighted)" python 7_stance_by_usertype_dedup.py
            ;;
        8)
            echo -e "${YELLOW}=== 🖼️  Step 8: Final Visualizations ===${RESET}"
            run_step "Country bar chart" python 8_country_bar_chart.py
            run_step "Keyword bar chart" python 8_keyword_bar_chart.py
            run_step "Split by userType" python 8_split_by_usertype.py
            run_step "Plot duplicates" python 8_plot_duplicates.py
            run_step "Length bin stance" python 8_Length_Bin_Stance.py
            run_step "Submission timing" python 8_submission_timing.py
            ;;
        9)
            echo -e "${BLUE}▶ Running full pipeline (safe mode)...${RESET}"

            echo -e "${YELLOW}=== 2️⃣  Deduplication & Cleaning ===${RESET}"
            run_step "Count characters" python 2_count_characters.py
            run_step "Feedback similarity matrix" python 2_feedback_similarity_matrix.py
            run_step "Split duplicate vs unique" python 2_split_duplicate_unique.py
            run_step "Compare attachment texts" python 2_compare_attachment_texts.py
            run_step "Attachment similarity matrix" python 2_attachments_similarity_matrix.py
            run_step "Translate non-English attachments" python 2_translate_attachments.py

            echo -e "${YELLOW}=== 3️⃣  Stance Detection ===${RESET}"
            run_step "Stance detection" python 3_stance_detection.py
            run_step "Log metrics" python 3_log_metrics.py
            # SKIP: 3_manage_zero_shot.py
            # SKIP: 3_reset_supervised.py

            echo -e "${YELLOW}=== 4️⃣  Similarity Analysis ===${RESET}"
            run_step "Feedback similarity analysis" python 4_similarity_analysis.py
            run_step "Attachment similarity analysis" python 4_similarity_analysis_attachments.py

            echo -e "${YELLOW}=== 5️⃣  Bot Detection ===${RESET}"
            run_step "Bot detection" python 5_bot_detection_v4.py
            run_step "Bot detection analysis" python 5_bot_detection_v4_analysis.py
            run_step "Bot detection analysis v4.1" python 5_bot_detection_v4.1_analysis.py

            echo -e "${YELLOW}=== 6️⃣  Content Analysis ===${RESET}"
            run_step "Submission volume by day" python 6_submission_volume_by_day.py
            run_step "Word frequency counter" python 6_word_frequency_counter.py
            run_step "Bigram analysis" python 6_bigrams.py
            run_step "Country summary" python 6_country.py
            run_step "Keyword counter" python 6_keyword_counter.py

            echo -e "${YELLOW}=== 7️⃣  Stance Visualizations ===${RESET}"
            run_step "Stance analysis" python 7_stance_analysis.py
            run_step "Stance by usertype" python 7_stance_by_usertype.py
            run_step "Stance by usertype (dedup weighted)" python 7_stance_by_usertype_dedup.py

            echo -e "${YELLOW}=== 8️⃣  Final Visualizations ===${RESET}"
            run_step "Country bar chart" python 8_country_bar_chart.py
            run_step "Keyword bar chart" python 8_keyword_bar_chart.py
            run_step "Split by userType" python 8_split_by_usertype.py
            run_step "Plot duplicates" python 8_plot_duplicates.py
            run_step "Length bin stance" python 8_Length_Bin_Stance.py
            run_step "Submission timing" python 8_submission_timing.py

            echo -e "${YELLOW}=== 🗄️  Load database ===${RESET}"
            run_step "Load SQLite DB" python load_db.py

            echo -e "${GREEN}✅ Full pipeline complete (safe mode).${RESET}"
            ;;
        a)
            echo -e "${YELLOW}=== 🏷️  Rerun stance detection + metrics ===${RESET}"
            run_step "Stance detection" python 3_stance_detection.py
            run_step "Log metrics" python 3_log_metrics.py
            ;;
        n)
            echo -e "${YELLOW}=== 🌙 Nightly run: stance → downstream ===${RESET}"
            echo -e "(Stage 4 skipped — similarity matrices do not depend on stance)\n"

            run_step "Stance detection" python 3_stance_detection.py
            run_step "Log metrics" python 3_log_metrics.py

            echo -e "${YELLOW}--- Stage 5: Bot Detection ---${RESET}"
            run_step "Bot detection" python 5_bot_detection_v4.py
            run_step "Bot detection analysis" python 5_bot_detection_v4_analysis.py
            run_step "Bot detection analysis v4.1" python 5_bot_detection_v4.1_analysis.py

            echo -e "${YELLOW}--- Stage 6: Content Analysis ---${RESET}"
            run_step "Submission volume by day" python 6_submission_volume_by_day.py
            run_step "Word frequency counter" python 6_word_frequency_counter.py
            run_step "Bigram analysis" python 6_bigrams.py
            run_step "Country summary" python 6_country.py
            run_step "Keyword counter" python 6_keyword_counter.py

            echo -e "${YELLOW}--- Stage 7: Stance Visualizations ---${RESET}"
            run_step "Stance analysis" python 7_stance_analysis.py
            run_step "Stance by usertype" python 7_stance_by_usertype.py
            run_step "Stance by usertype (dedup weighted)" python 7_stance_by_usertype_dedup.py

            echo -e "${YELLOW}--- Stage 8: Final Visualizations ---${RESET}"
            run_step "Country bar chart" python 8_country_bar_chart.py
            run_step "Keyword bar chart" python 8_keyword_bar_chart.py
            run_step "Split by userType" python 8_split_by_usertype.py
            run_step "Plot duplicates" python 8_plot_duplicates.py
            run_step "Length bin stance" python 8_Length_Bin_Stance.py
            run_step "Submission timing" python 8_submission_timing.py

            echo -e "${GREEN}✅ Nightly run complete.${RESET}"
            ;;
        p)
            echo -e "${YELLOW}=== 🎨 Regenerate plots (5 + 6 + 7 + 8) ===${RESET}"
            echo -e "(Stage 4 skipped — use option 4 to regenerate similarity plots)\n"

            echo -e "${YELLOW}--- Stage 5: Bot Detection ---${RESET}"
            run_step "Bot detection" python 5_bot_detection_v4.py
            run_step "Bot detection analysis" python 5_bot_detection_v4_analysis.py
            run_step "Bot detection analysis v4.1" python 5_bot_detection_v4.1_analysis.py

            echo -e "${YELLOW}--- Stage 6: Content Analysis ---${RESET}"
            run_step "Submission volume by day" python 6_submission_volume_by_day.py
            run_step "Word frequency counter" python 6_word_frequency_counter.py
            run_step "Bigram analysis" python 6_bigrams.py
            run_step "Country summary" python 6_country.py
            run_step "Keyword counter" python 6_keyword_counter.py

            echo -e "${YELLOW}--- Stage 7: Stance Visualizations ---${RESET}"
            run_step "Stance analysis" python 7_stance_analysis.py
            run_step "Stance by usertype" python 7_stance_by_usertype.py
            run_step "Stance by usertype (dedup weighted)" python 7_stance_by_usertype_dedup.py

            echo -e "${YELLOW}--- Stage 8: Final Visualizations ---${RESET}"
            run_step "Country bar chart" python 8_country_bar_chart.py
            run_step "Keyword bar chart" python 8_keyword_bar_chart.py
            run_step "Split by userType" python 8_split_by_usertype.py
            run_step "Plot duplicates" python 8_plot_duplicates.py
            run_step "Length bin stance" python 8_Length_Bin_Stance.py
            run_step "Submission timing" python 8_submission_timing.py

            echo -e "${GREEN}✅ All plots regenerated.${RESET}"
            ;;
        b)
            echo -e "${YELLOW}=== 💾 Backup ===${RESET}"
            echo "1) Backup input files (feedback_ids, feedback_details, attachments CSVs) — replaces previous"
            echo "2) Backup pipeline CSVs — compressed, rolling set"
            echo "3) Both"
            read -p "Choose [1/2/3]: " backup_choice

            BACKUP_DIR="data/backups"
            INPUT_BACKUP="$BACKUP_DIR/inputs"
            PIPELINE_BACKUP="$BACKUP_DIR/pipeline"
            PIPELINE_COPIES=$(python -c "from settings import PIPELINE_BACKUP_COPIES; print(PIPELINE_BACKUP_COPIES)")
            TS=$(date +%Y-%m-%d_%H%M%S)

            do_input_backup() {
                echo -e "${BLUE}▶ Backing up input files...${RESET}"
                mkdir -p "$INPUT_BACKUP"
                for f in feedback_ids.csv feedback_details.csv attachment_unique.csv attachment_duplicates.csv; do
                    [ -f "data/$f" ] && cp "data/$f" "$INPUT_BACKUP/$f" && echo "  ✔ $f"
                done
                echo -e "${GREEN}✔ Input backup complete → $INPUT_BACKUP${RESET}\n"
            }

            do_pipeline_backup() {
                echo -e "${BLUE}▶ Backing up pipeline CSVs (compressed)...${RESET}"
                mkdir -p "$PIPELINE_BACKUP"
                ARCHIVE="$PIPELINE_BACKUP/$TS.tar.gz"
                tar -czf "$ARCHIVE" \
                    --exclude="data/backups" \
                    --exclude="data/stance_detection/supervised_model" \
                    --exclude="data/stance_detection/model_backups" \
                    --exclude="data/stance_detection/supervised_model_*" \
                    data/ 2>/dev/null
                echo -e "${GREEN}✔ Pipeline backup → $ARCHIVE${RESET}"
                # Prune oldest beyond PIPELINE_BACKUP_COPIES
                ARCHIVES=($(ls -1t "$PIPELINE_BACKUP"/*.tar.gz 2>/dev/null))
                COUNT=${#ARCHIVES[@]}
                if [ "$COUNT" -gt "$PIPELINE_COPIES" ]; then
                    for i in "${ARCHIVES[@]:$PIPELINE_COPIES}"; do
                        rm "$i" && echo "  🗑️  Pruned $i"
                    done
                fi
                echo -e "${GREEN}✔ Kept $PIPELINE_COPIES most recent pipeline backups${RESET}\n"
            }

            case $backup_choice in
                1) do_input_backup ;;
                2) do_pipeline_backup ;;
                3) do_input_backup; do_pipeline_backup ;;
                *) echo "Invalid choice, skipping." ;;
            esac
            ;;

        d)
            echo -e "${YELLOW}=== 🗄️  Load SQLite database ===${RESET}"
            run_step "Load SQLite DB" python load_db.py
            ;;
        t)
            echo -e "${YELLOW}=== ⚗️  Test stance run (no production files modified) ===${RESET}"
            STANCE_TEST_RUN=true run_step "Test stance detection" python 3_stance_detection.py
            ;;

        0)
            echo -e "${GREEN}✅ Exiting pipeline menu.${RESET}"
            exit 0
            ;;
    esac
done
