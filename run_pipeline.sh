#!/bin/bash
set -e  # stop on error

# 🎨 Colors
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
RESET="\033[0m"

echo -e "${GREEN}🚀 EU Feedback Analysis Pipeline Menu${RESET}"

# Helper to run a step
run_step() {
    echo -e "${BLUE}▶ Running: $1${RESET}"
    shift
    "$@"
    echo -e "${GREEN}✔ Completed: $1${RESET}\n"
}

# Step 0: Ask for IDs once at the start
echo -e "${YELLOW}=== 0️⃣ Setup IDs ===${RESET}"
read -p "Enter PUBLICATION_ID (default 20315): " PUBLICATION_ID
PUBLICATION_ID=${PUBLICATION_ID:-20315}

read -p "Enter INITIATIVE_ID (default 12645): " INITIATIVE_ID
INITIATIVE_ID=${INITIATIVE_ID:-12645}

echo -e "${GREEN}✔ Using PUBLICATION_ID=$PUBLICATION_ID INITIATIVE_ID=$INITIATIVE_ID${RESET}\n"

while true; do
    echo -e "${YELLOW}=== Choose a pipeline stage ===${RESET}"
    echo "1) Fetch & Collect Raw Data"
    echo "2) Deduplication & Cleaning"
    echo "3) Similarity & Group Analysis"
    echo "4) Bot Detection"
    echo "5) Stats & Keyword Analysis"
    echo "6) Stance Detection & Evaluation"
    echo "7) Feedback Visualizations"
    echo "8) Run ALL stages (safe mode)"
    echo "9) Exit"
    read -p "Enter choice [1-9]: " choice

    case $choice in
        1)
            echo -e "${YELLOW}=== 📥 Step 1: Fetch & Collect Raw Data ===${RESET}"
            run_step "Collect feedback IDs" \
                python 1_get_feedback_ids.py --publication-id "$PUBLICATION_ID"

            run_step "Fetch feedback details (retry)" \
                python 1_fetch_feedback_details_with_retry.py --ids-file data/feedback_ids.csv --output data/feedback_details.csv --initiative-id "$INITIATIVE_ID"

            run_step "Download attachments" \
                python 1_download_attachments.py --input data/feedback_details.csv --output data/attachments/

            run_step "Check missing IDs" \
                python 1_look_missing_ids.py --ids-file data/feedback_ids.csv --details-file data/feedback_details.csv
            ;;
        2)
            echo -e "${YELLOW}=== 🧹 Step 2: Deduplication & Cleaning ===${RESET}"
            run_step "Count characters" python 2_count_characters.py
            run_step "Split duplicate vs unique" python 2_split_duplicate_unique.py
            run_step "Compare attachment texts" python 2_compare_attachment_texts.py
            run_step "Feedback similarity matrix" python 2_feedback_similarity_matrix.py
            run_step "Attachment similarity matrix" python 2_attachments_similarity_matrix.py
            ;;
        3)
            echo -e "${YELLOW}=== 🧠 Step 3: Similarity & Group Analysis ===${RESET}"
            run_step "Feedback similarity analysis" python 3_similarity_analysis.py
            run_step "Attachment similarity analysis" python 3_similarity_analysis_attachments.py
            ;;
        4)
            echo -e "${YELLOW}=== 🤖 Step 4: Bot Detection ===${RESET}"
            run_step "Bot detection" python 4_bot_detection_v2.py
            ;;
        5)
            echo -e "${YELLOW}=== 📊 Step 5: Stats & Keyword Analysis ===${RESET}"
            run_step "Keyword counter" python 5_keyword_counter.py
            run_step "Word frequency counter" python 5_word_frequency_counter.py
            run_step "Bigram analysis" python 5_bigrams.py
            run_step "Country summary" python 5_country.py
            ;;
        6)
            echo -e "${YELLOW}=== 🧭 Step 6: Stance Detection & Evaluation ===${RESET}"
            run_step "Stance detection" python 6_stance_detection.py
            run_step "Stance analysis" python 6_stance_analysis.py
            run_step "Log metrics" python 6_log_metrics.py

            read -p "⚠️ Run manage_zero_shot.py? (y/N): " confirm_zero
            if [[ "$confirm_zero" == "y" || "$confirm_zero" == "Y" ]]; then
                run_step "Manage zero-shot runs" python 6_manage_zero_shot.py
            else
                echo "⏩ Skipped manage_zero_shot.py"
            fi

            read -p "⚠️ Run reset_supervised.py (this will wipe supervised stance)? (y/N): " confirm_reset
            if [[ "$confirm_reset" == "y" || "$confirm_reset" == "Y" ]]; then
                run_step "Reset supervised pipeline" python 6_reset_supervised.py
            else
                echo "⏩ Skipped reset_supervised.py"
            fi
            ;;
        7)
            echo -e "${YELLOW}=== 📈 Step 7: Feedback Visualizations ===${RESET}"
            run_step "Country bar chart" python 7_country_bar_chart.py
            run_step "Keyword bar chart" python 7_keyword_bar_chart.py
            run_step "Split by userType" python 7_split_by_usertype.py
            run_step "Plot duplicates" python 7_plot_duplicates.py
            run_step "Length bin stance" python 7_Length_Bin_Stance.py
            ;;
        8)
            echo -e "${BLUE}▶ Running full pipeline (safe mode)...${RESET}"
            # Run stages 1–7, but skip manage_zero_shot and reset_supervised
            echo -e "${YELLOW}=== 1️⃣ Fetch & Collect Raw Data ===${RESET}"

            if [ -f "data/feedback_details.csv" ] && [ "$FORCE" != "true" ]; then
                echo -e "${GREEN}✔ feedback_details.csv already exists, skipping Step 1 (use FORCE=true to redownload).${RESET}\n"
            else
                run_step "Collect feedback IDs" \
                    python 1_get_feedback_ids.py --publication-id "$PUBLICATION_ID"

                run_step "Fetch feedback details (retry)" \
                    python 1_fetch_feedback_details_with_retry.py --ids-file data/feedback_ids.csv --output data/feedback_details.csv --initiative-id "$INITIATIVE_ID"

                run_step "Download attachments" \
                    python 1_download_attachments.py --input data/feedback_details.csv --output data/attachments/

                run_step "Check missing IDs" \
                    python 1_look_missing_ids.py --ids-file data/feedback_ids.csv --details-file data/feedback_details.csv
            fi

            echo -e "${YELLOW}=== 🧹 Step 2: Deduplication & Cleaning ===${RESET}"
            run_step "Count characters" python 2_count_characters.py
            run_step "Split duplicate vs unique" python 2_split_duplicate_unique.py
            run_step "Compare attachment texts" python 2_compare_attachment_texts.py
            run_step "Feedback similarity matrix" python 2_feedback_similarity_matrix.py
            run_step "Attachment similarity matrix" python 2_attachments_similarity_matrix.py

            echo -e "${YELLOW}=== 🧠 Step 3: Similarity & Group Analysis ===${RESET}"
            run_step "Feedback similarity analysis" python 3_similarity_analysis.py
            run_step "Attachment similarity analysis" python 3_similarity_analysis_attachments.py

            echo -e "${YELLOW}=== 🤖 Step 4: Bot Detection ===${RESET}"
            run_step "Bot detection" python 4_bot_detection_v2.py

            echo -e "${YELLOW}=== 📊 Step 5: Stats & Keyword Analysis ===${RESET}"
            run_step "Keyword counter" python 5_keyword_counter.py
            run_step "Word frequency counter" python 5_word_frequency_counter.py
            run_step "Bigram analysis" python 5_bigrams.py
            run_step "Country summary" python 5_country.py

            echo -e "${YELLOW}=== 🧭 Step 6: Stance Detection & Evaluation ===${RESET}"
            run_step "Stance detection" python 6_stance_detection.py
            run_step "Stance analysis" python 6_stance_analysis.py
            run_step "Log metrics" python 6_log_metrics.py
            # SKIP: 6_manage_zero_shot.py
            # SKIP: 6_reset_supervised.py

            echo -e "${YELLOW}=== 📈 Step 7: Feedback Visualizations ===${RESET}"
            run_step "Country bar chart" python 7_country_bar_chart.py
            run_step "Keyword bar chart" python 7_keyword_bar_chart.py
            run_step "Split by userType" python 7_split_by_usertype.py
            run_step "Plot duplicates" python 7_plot_duplicates.py
            run_step "Length bin stance" python 7_Length_Bin_Stance.py

            echo -e "${GREEN}✅ Full pipeline complete (safe mode).${RESET}"
            ;;
        9)
            echo -e "${GREEN}✅ Exiting pipeline menu.${RESET}"
