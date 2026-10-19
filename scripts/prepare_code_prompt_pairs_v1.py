"""Prepare source associations across the complete admitted public code pack."""
from pathlib import Path
import json

from slm_lab.code_prompt_pair_corpus_v1 import prepare

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    print(json.dumps(prepare(ROOT/'data/codenet-durable-pack-v2',ROOT/'data/code-prompt-pairs-v1'),indent=2))
