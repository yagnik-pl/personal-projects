.PHONY: install test download-data baseline layer-analysis adaptive fixed-depth compare figures all-experiments clean help

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

download-data:
	python scripts/download_data.py --datasets scifact nfcorpus

baseline:
	python scripts/run_baseline.py --config configs/baseline.yaml

layer-analysis:
	python scripts/run_layer_analysis.py --config configs/layer_analysis.yaml

adaptive:
	python scripts/run_adaptive.py --config configs/adaptive.yaml

fixed-depth:
	python scripts/run_fixed_depth.py --config configs/adaptive.yaml

compare:
	python scripts/compare_results.py

figures:
	python scripts/generate_figures.py

all-experiments: baseline layer-analysis fixed-depth adaptive compare figures

clean:
	rm -rf results/ data/ __pycache__

help:
	@echo "Available targets:"
	@echo "  install          : Install dependencies"
	@echo "  test             : Run tests"
	@echo "  download-data    : Download datasets"
	@echo "  baseline         : Run baseline experiments"
	@echo "  layer-analysis   : Run layer analysis experiments"
	@echo "  adaptive         : Run adaptive experiments"
	@echo "  fixed-depth      : Run fixed-depth experiments"
	@echo "  compare          : Compare results"
	@echo "  figures          : Generate figures"
	@echo "  all-experiments  : Run all experiments (except download-data)"
	@echo "  clean            : Clean generated files and results"
	@echo "  help             : Show this help message"
