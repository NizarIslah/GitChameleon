input_dir=rag_results/
python scripts/annotate_csvs.py gemini --keys answer --out-dir gemini_answers
python scripts/self_debug_jsonl.py dataset/final_fix_dataset_hitrate.jsonl gemini_self_debug_answers_merged/ gemini_self_debug_answers_merged_correct/
python scripts/api_hitrate.py gemini_self_debug_answers_merged_correct/ gemini_debug_hitrate_correct/
# python scripts/pull_results_csv.py rag_results/ -c passed passed_manual