#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/slm-lab
while [[ ! -f data/public-reference-v1/manifest.json ]]; do
  sleep 15
done
.venv/bin/python -u scripts/prepare_broad_pilot.py
.venv/bin/python -u -m slm_lab.evaluation --checkpoint checkpoints/full-first-pretrain/model.pt --data data/broad-evaluation-v1/code.valid.jsonl --output reports/broad-code-baseline.json --device cuda --max-examples 128
.venv/bin/python -u -m slm_lab.evaluation --checkpoint checkpoints/full-first-pretrain/model.pt --data data/broad-evaluation-v1/domains.valid.jsonl --output reports/broad-domains-baseline.json --device cuda --max-examples 128
pilot_steps=$(.venv/bin/python -c 'import json; print(json.load(open("reports/broad-public-pilot-plan.json"))["optimizer_steps"])')
.venv/bin/python -u -m torch.distributed.run --standalone --nproc_per_node 4 -m slm_lab.training pretrain --config configs/model.json --tokenizer data/tokenizer-corpus/tokenizer.json --data data/broad-mixture-v1 --checkpoint checkpoints/full-first-pretrain/model.pt --output checkpoints/broad-public-pilot-v1 --device cuda --steps "$pilot_steps" --batch-size 1 --gradient-accumulation 16 --learning-rate 0.0003 --gradient-checkpointing --checkpoint-every 100 --keep-checkpoints 1 --log-every 10
.venv/bin/python -u -m slm_lab.evaluation --checkpoint checkpoints/broad-public-pilot-v1/model.pt --data data/broad-evaluation-v1/code.valid.jsonl --output reports/broad-code-after.json --device cuda --max-examples 128
.venv/bin/python -u -m slm_lab.evaluation --checkpoint checkpoints/broad-public-pilot-v1/model.pt --data data/broad-evaluation-v1/domains.valid.jsonl --output reports/broad-domains-after.json --device cuda --max-examples 128
.venv/bin/python -u -m slm_lab.evaluation --checkpoint checkpoints/broad-public-pilot-v1/model.pt --data data/broad-mixture-v1 --output reports/broad-mixture-after.json --device cuda --max-examples 64
.venv/bin/python -u -m slm_lab.code_evaluation --checkpoint checkpoints/broad-public-pilot-v1/model.pt --output reports/code-generation-after-pilot.json --device cuda
