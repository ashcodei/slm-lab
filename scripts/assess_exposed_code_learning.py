"""Distinguish exact exposed-program learning from held-task generalization."""
from datetime import datetime, timezone
import gc
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from slm_lab.code_quality import execute, instruction_code
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.verified_code_diagnostics_v2 import evaluate, select

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = ROOT / 'reports/exposed-code-learning-after-instruction-v1'
    if reports.exists():
        raise ValueError('Preserve an existing exposed-source diagnostic')
    reports.mkdir(); torch.set_num_threads(1)
    trial = ROOT / 'reports/instruction-generalization-probe-v3'
    if json.loads((trial / 'status.json').read_text())['status'] != 'balanced public instruction learning and assessments completed; broad capability unverified':
        raise ValueError('Require the completed exact instruction trial')
    plan = json.loads((trial / 'plan.json').read_text()); trained = json.loads((trial / 'training.json').read_text())
    data = Path(plan['input_manifest']).parent / 'verified_python.jsonl'; exposures = data.parent / 'exposures.jsonl'
    manifest = json.loads(Path(plan['input_manifest']).read_text())
    for path in [data, exposures]:
        if file_hash(path) != manifest['files_sha256'][path.name]:
            raise ValueError('Exact completed instruction source/exposure bytes differ')
    rows = list(jsonl(data)); seen = {row['id'] for row in jsonl(exposures) if row['pool'] == 'verified_python'}
    tokenizer = Tokenizer.from_file(str(Path(plan['base_checkpoint']).parent / 'tokenizer.json'))
    selected = select([row for row in rows if row['id'] in seen], tokenizer, 128)
    verified = []
    for row in selected:
        actual = execute({'language': 'Python', 'code': instruction_code(row['response']), 'function': row['function'],
                          'cases': row['verification_cases'], 'stress': []})
        if not actual.get('functional_passed'):
            raise ValueError('Exposed public positive target fails its original examples: ' + row['id'])
        verified.append({'id': row['id'], 'function': row['function'], 'original_program_sha256': row['original_program_sha256'],
                         'published_cases': len(row['verification_cases']), 'result': actual})
    atomic_json(reports / 'source-verification.json', {'source_data_sha256': file_hash(data), 'actual_exposures_sha256': file_hash(exposures),
        'all128_original_positive_targets_pass_published_examples': True, 'source_examples': verified,
        'exposed_original_algorithms': len({row['original_program_sha256'] for row in rows if row['id'] in seen}),
        'available_original_algorithms': len({row['original_program_sha256'] for row in rows}), 'goal_achieved': False})
    checkpoints = [Path(plan['base_checkpoint']), Path(trained['candidate_checkpoint'])]
    expected = [plan['base_checkpoint_sha256'], trained['candidate_checkpoint_sha256']]
    for label, checkpoint, digest in zip(['before', 'after'], checkpoints, expected):
        if file_hash(checkpoint) != digest:
            raise ValueError('Own baseline/candidate checkpoint differs')
        atomic_json(reports / 'status.json', {'status': 'assessing actually exposed public source programs', 'policy': label,
            'updated_at': datetime.now(timezone.utc).isoformat(), 'examples': 128, 'goal_achieved': False})
        torch.cuda.set_per_process_memory_fraction(.2, 3)
        evaluate(checkpoint, data, reports / (label + '.json'), device='cuda:3', count=128, max_new_tokens=512, exposures=exposures)
        gc.collect(); torch.cuda.empty_cache()
    before = json.loads((reports / 'before.json').read_text()); after = json.loads((reports / 'after.json').read_text())
    for field in ['evaluation_provenance', 'data_sha256', 'actual_exposures_sha256', 'selection', 'generation_budget',
                  'tokenizer_sha256', 'inference_precision', 'inference_device_type']:
        if before[field] != after[field]:
            raise ValueError('Unmatched exposed-program comparison: ' + field)
    fields = ['id', 'prompt_sha256', 'original_program_sha256', 'published_cases', 'target_tokens']
    identity = lambda report: [tuple(row[field] for field in fields) for row in report['results']]
    if identity(before) != identity(after) or len(set(identity(before))) != 128:
        raise ValueError('Changed, duplicate or incomplete exposed-source population')
    metrics = ['teacher_forced_nll', 'teacher_forced_token_accuracy', 'syntax_pass_rate', 'published_example_functional_rate']
    atomic_json(reports / 'comparison.json', {'completed_at': datetime.now(timezone.utc).isoformat(),
        'metrics': {name: {'before': before[name], 'after': after[name]} for name in metrics},
        'base_checkpoint_sha256': expected[0], 'candidate_checkpoint_sha256': expected[1],
        'previously_correct_sources_lost': [a['id'] for a, b in zip(before['results'], after['results']) if a['candidate']['functional_passed'] and not b['candidate']['functional_passed']],
        'newly_correct_sources': [b['id'] for a, b in zip(before['results'], after['results']) if not a['candidate']['functional_passed'] and b['candidate']['functional_passed']],
        'scope': 'Actually exposed public original-name training programs with visible published examples. Learning/memorization diagnostic, not held-out or efficiency evidence.',
        'automatic_policy_promotion': False, 'goal_achieved': False})
    atomic_json(reports / 'status.json', {'status': 'exposed-source learning comparison completed; broader capability unverified',
        'updated_at': datetime.now(timezone.utc).isoformat(), 'goal_achieved': False})


if __name__ == '__main__':
    main()
