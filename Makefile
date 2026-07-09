.PHONY: experiment1 experiment1-smoke experiment1-real-gpu

experiment1:
	python scripts/run_experiment1.py --allow-provisional-edits

experiment1-smoke: experiment1
	python scripts/simulate_experiment1_detection.py

experiment1-real-gpu:
	scripts/run_experiment1_real_gpu.sh
