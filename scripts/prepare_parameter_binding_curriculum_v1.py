"""Prepare checked public argument-name variants without altering live training."""
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.model import ModelConfig
from slm_lab.parameter_binding_curriculum_v1 import prepare
from slm_lab.training import Encoder

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    reports = ROOT / 'reports/parameter-binding-curriculum-v1'
    if reports.exists(): raise ValueError('Inspect earlier preparation; use a fresh generation')
    reports.mkdir()
    files = [ROOT / name for name in capture()['files_sha256']]
    files += [Path(__file__), ROOT / 'src/slm_lab/parameter_binding_curriculum_v1.py',
              ROOT / 'src/slm_lab/instruction_generalization_trial_v3.py']
    runtime = capture(files)
    source = ROOT / 'checkpoints/instruction-generalization-probe-v3/data/verified_python.jsonl'
    tokenizer = ROOT / 'checkpoints/instruction-generalization-probe-v3/candidate/tokenizer.json'
    config = ROOT / 'configs/model.json'
    inputs = [source, source.parent / 'manifest.json', tokenizer, config,
              ROOT / 'reports/instruction-generalization-probe-v3/plan.json']
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in inputs}
    plan = json.loads(inputs[-1].read_text())
    if file_hash(tokenizer) != plan['tokenizer_sha256']:
        raise ValueError('Own source tokenizer differs')
    def status(message, **fields):
        atomic_json(reports / 'status.json', {'status': message,
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'trained': False, 'goal_achieved': False, **fields})
    try:
        status('preparing and reexecuting public parameter-binding variants')
        encoder = Encoder(Tokenizer.from_file(str(tokenizer)), ModelConfig.load(config), source.parent)
        result = prepare(source, ROOT / 'data/parameter-binding-curriculum-v1', encoder, workers=4)
        if capture(files) != runtime or any(file_hash(ROOT / name) != digest for name, digest in hashes.items()):
            raise ValueError('Bound source/runtime changed during preparation')
        result.update(tokenizer_sha256=file_hash(tokenizer), model_config_sha256=file_hash(config),
                      preparation_runtime=runtime, source_inputs_sha256=hashes)
        atomic_json(ROOT / 'data/parameter-binding-curriculum-v1/manifest.json', result)
        atomic_json(reports / 'preparation.json', result)
        status('public parameter-binding preparation completed; learning untested',
               transformed_training_rows=result['transformed_training_rows'],
               distinct_original_algorithms=result['distinct_original_algorithms'])
        print(json.dumps({key: value for key, value in result.items()
                          if key not in {'preparation_runtime', 'source_inputs_sha256'}}, indent=2))
    except Exception as error:
        status('public parameter-binding preparation stopped; requires inspection', error=str(error))
        raise


if __name__ == '__main__': main()
