input_dir=rag_results/
python scripts/annotate_csvs.py rag_results --keys answer --out-dir rag_results_answers
python scripts/self_debug_jsonl.py dataset/final_fix_dataset.jsonl rag_results_answers/ rag_results_answers/
python scripts/api_hitrate.py rag_results_answers/ rag_results_hitrate/
# python scripts/pull_results_csv.py rag_results/ -c passed passed_manual
# python scripts/retrieval_recall.py --gt_file dataset/final_fix_dataset.jsonl --pred_dir rag_results/