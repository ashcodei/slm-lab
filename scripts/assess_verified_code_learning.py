"""Matched before/after seen-program diagnosis of the completed own code probe."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/verified-code-learning-diagnostic-v1'


def main():
    REPORTS.mkdir(exist_ok=True)
    if any(REPORTS.iterdir()):
        raise ValueError('Existing diagnostic requires inspection; do not overwrite')
    def status(message, **fields):
        atomic_json(REPORTS / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
                                            'goal_achieved': False, **fields})
    try:
        probe = ROOT / 'reports/verified-code-foundation-probe-v1'
        plan = json.loads((probe / 'plan.json').read_text())
        if json.loads((probe / 'status.json').read_text())['status'] != 'verified-code intervention and local before/after checks completed; broader capability unverified':
            raise ValueError('Original coding probe has not completed')
        checkpoints = ROOT / 'checkpoints/verified-code-foundation-probe-v1'
        data = ROOT / 'data/verified-code-curriculum-v1/train.jsonl'
        if file_hash(data) != plan['training_data_sha256']:
            raise ValueError('Diagnostic data differ from actual code training')
        base = checkpoints / 'base/model.pt'; candidate = checkpoints / 'candidate/model.pt'
        if file_hash(base) != plan['base_checkpoint_sha256']:
            raise ValueError('Frozen own baseline checkpoint differs')
        state = torch.load(candidate, map_location='cpu', weights_only=True, mmap=True)
        if state['steps'] != plan['steps'] or state['stage'] != 'sft' or state['data_sha256'] != file_hash(data):
            raise ValueError('Candidate differs from the completed own intervention')
        del state
        atomic_json(REPORTS / 'plan.json', {'original_probe_plan_sha256': file_hash(probe / 'plan.json'),
                    'baseline_sha256': file_hash(base), 'candidate_sha256': file_hash(candidate),
                    'data_sha256': file_hash(data), 'examples': 32, 'generation_budget': 512,
                    'physical_gpu': 1, 'goal_achieved': False,
                    'scope': 'Actual short seen-training examples; does not establish generalization or efficiency.'})
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '1', 'OMP_NUM_THREADS': '1',
                       'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}
        results = {}
        for branch, checkpoint in [('baseline', base), ('candidate', candidate)]:
            status('assessing seen verified training programs', branch=branch)
            output = REPORTS / (branch + '.json')
            subprocess.run([sys.executable, '-u', '-m', 'slm_lab.verified_code_diagnostics',
                            '--checkpoint', str(checkpoint), '--data', str(data), '--output', str(output),
                            '--device', 'cuda', '--examples', '32', '--max-new-tokens', '512'],
                           cwd=ROOT, env=environment, check=True)
            results[branch] = json.loads(output.read_text())
        first, second = results['baseline'], results['candidate']
        for key in ['data_sha256', 'source_manifest_sha256', 'evaluation_provenance', 'selection',
                    'examples', 'generation_budget', 'inference_precision', 'tokenizer_sha256']:
            if first[key] != second[key]:
                raise ValueError('Seen-program comparison protocol differs: ' + key)
        fields = ['id', 'function', 'original_program_sha256', 'prompt_sha256', 'source_response_tokens', 'published_cases']
        if [[row[key] for key in fields] for row in first['results']] != [[row[key] for key in fields] for row in second['results']]:
            raise ValueError('Seen-program diagnostic cases differ')
        metrics = ['teacher_forced_nll', 'teacher_forced_token_accuracy', 'syntax_pass_rate', 'published_example_functional_rate']
        atomic_json(REPORTS / 'comparison.json', {'results': {key: {'baseline': first[key], 'candidate': second[key]} for key in metrics},
                    'examples': first['examples'], 'goal_achieved': False, 'production_promoted': False,
                    'limitations': second['limitations']})
        status('seen-program learning diagnostic completed; held-out capability remains unverified')
    except Exception as error:
        status('seen-program diagnostic stopped; requires inspection', error=str(error))
        raise


if __name__ == '__main__':
    main()
