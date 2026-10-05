.PHONY: install test prepare loso ablations

install:
	python -m pip install -e .

test:
	PYTHONPATH=src pytest -q

prepare:
	python scripts/prepare_data.py --config configs/default.yaml

loso:
	python scripts/train_loso.py --config configs/default.yaml

ablations:
	python scripts/run_ablations.py --config configs/default.yaml
