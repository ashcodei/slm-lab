"""Supply exact native evaluator sidecars once the owned arm folders exist.

The v2 controller creates model/resume files but omits the tokenizer sidecar
expected by existing native evaluators. This output-only helper changes no
running source, neural weight, input sequence, model option or scoring rule.
"""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exclusive_inference_resources_v1 import identity

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/own-onpolicy-pilot-v2'
OUTPUT = ROOT / 'checkpoints/own-onpolicy-pilot-v2'


def main():
    receipt = REPORTS / 'native-tokenizer-sidecars.json'
    if receipt.exists():
        raise ValueError('Preserve the original sidecar-preparation receipt')
    plan = json.loads((REPORTS / 'plan.json').read_text())
    def live_owner():
        actual = identity(plan['owner']['pid'])
        return actual and actual['ticks'] == plan['owner']['ticks'] and actual['state'] not in {'Z', 'X'}
    if not live_owner() or file_hash(plan['tokenizer']) != plan['tokenizer_sha256']:
        raise ValueError('Exact live original experiment and own tokenizer required')
    with (REPORTS / 'children.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        actors = json.loads((REPORTS / 'children.json').read_text())
        actors.append(dict(**identity(os.getpid()), subject='prepare-identical-native-tokenizer-sidecars'))
        atomic_json(REPORTS / 'children.json', actors)
    completed = []
    while len(completed) != len(plan['arms']):
        if not live_owner() or (REPORTS / 'restored.json').exists() or time.time() > plan['expires_at_unix']:
            raise ValueError('Owned experiment ended before native sidecar preparation; do not create replacement work')
        for arm in plan['arms']:
            if arm in completed:
                continue
            folder = OUTPUT / arm
            if folder.is_dir() and not folder.is_symlink():
                destination = folder / 'tokenizer.json'
                if destination.exists():
                    raise ValueError('Inspect existing native tokenizer rather than overwriting it')
                shutil.copyfile(plan['tokenizer'], destination)
                if file_hash(destination) != plan['tokenizer_sha256']:
                    raise ValueError('Exact byte-identical own tokenizer sidecar required')
                completed.append(arm)
        if len(completed) != len(plan['arms']):
            time.sleep(.5)
    atomic_json(receipt, dict(completed_at=datetime.now(timezone.utc).isoformat(), arms=completed,
        tokenizer_sha256=plan['tokenizer_sha256'], source_helper=str(Path(__file__).resolve()),
        source_helper_sha256=file_hash(Path(__file__).resolve()),
        reason='Existing native evaluators read checkpoint.parent/tokenizer.json; version2controller omits these output sidecars. Exact original tokenizer copied before evaluation; running sources/weights/data/scoring unchanged.',
        model_updates=0, goal_achieved=False))
    print(json.dumps(dict(completed=completed, tokenizer_sha256=plan['tokenizer_sha256'])), flush=True)


if __name__ == '__main__':
    main()
