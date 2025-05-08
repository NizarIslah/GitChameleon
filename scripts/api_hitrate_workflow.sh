input_dir=cli_assistant_results/
python scripts/annotate_csvs.py cli_assistant_results --keys answer --out-dir cli_assistant_results_answers
python scripts/self_debug_jsonl.py dataset/final_fix_dataset.jsonl cli_assistant_results_answers/ cli_assistant_results_answers/
python scripts/api_hitrate.py cli_assistant_results_answers/ cli_assistant_results_hitrate/
# python scripts/pull_results_csv.py cli_assistant_results/ -c passed passed_manual