"""Produce a validated comparison after the exact instruction-trial actor ends."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation_provenance import capture
from slm_lab.instruction_trial_comparison import compare_reports
from slm_lab.model import ModelConfig, TinyMultimodalLM

ROOT = Path(__file__).resolve().parents[1]
DONE = 'balanced public instruction learning and assessments completed; broad capability unverified'
FILES = {'code': 'code-completion', 'code_instruction': 'code-instruction', 'cpp': 'cpp',
         'knowledge': 'knowledge', 'module_loss': 'module-loss'}


def alive(pid, ticks):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] == ticks and fields[0] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, default=ROOT / 'reports/instruction-generalization-probe-v3')
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--start-ticks')
    args = parser.parse_args(); folder = args.reports.resolve(); output = folder / 'comparison.json'
    source_files = [Path(__file__), ROOT / 'src/slm_lab/instruction_trial_comparison.py',
                    ROOT / 'src/slm_lab/alignment_selection.py', ROOT / 'src/slm_lab/code_tasks.py']
    comparison_runtime = capture(source_files)
    if output.exists():
        raise ValueError('Preserve completed instruction comparison evidence')
    if args.wait_pid is not None:
        if not args.start_ticks or not alive(args.wait_pid, args.start_ticks):
            raise ValueError('Expected producer is not verified live; inspect artifacts instead of assuming a wait')
        argv = Path(f'/proc/{args.wait_pid}/cmdline').read_bytes().split(b'\0')
        if not any(value.endswith(b'train_instruction_generalization_probe_v3.py') for value in argv):
            raise ValueError('Expected producer is not the isolated instruction trial')
        atomic_json(folder / 'comparison-status.json', {'status': 'waiting for exact live trial and all matched after-assessments',
            'pid': args.wait_pid, 'start_ticks': args.start_ticks, 'updated_at': datetime.now(timezone.utc).isoformat(), 'goal_achieved': False})
        while alive(args.wait_pid, args.start_ticks):
            time.sleep(15)
    status = json.loads((folder / 'status.json').read_text())
    if status['status'] != DONE:
        raise ValueError('Trial did not finish all declared learning and assessments: ' + status['status'])
    plan = json.loads((folder / 'plan.json').read_text()); training = json.loads((folder / 'training.json').read_text())
    if (training['completed_steps'] != plan['steps'] or training['steps'] != training['planned_steps']
            or training['actual_response_targets'] != plan['planned_actual_response_targets']
            or [row['step'] for row in training['metrics']] != list(range(1, plan['steps'] + 1))
            or sum(row['actual_target_tokens'] for row in training['metrics']) != training['actual_response_targets']):
        raise ValueError('Actual complete optimizer/target budget differs from the source plan')
    paths = [Path(plan['base_checkpoint']), Path(training['candidate_checkpoint'])]
    expected = [plan['base_checkpoint_sha256'], training['candidate_checkpoint_sha256']]
    if [file_hash(path) for path in paths] != expected:
        raise ValueError('Actual baseline/candidate checkpoint bytes differ from their reports')
    before = {key: json.loads((folder / ('before-' + name + '.json')).read_text()) for key, name in FILES.items()}
    after = {key: json.loads((folder / ('after-' + name + '.json')).read_text()) for key, name in FILES.items()}
    for reports, digest in zip([before, after], expected):
        if any(report['checkpoint_sha256'] != digest or report['evaluation_provenance'] != capture() for report in reports.values()):
            raise ValueError('Measured policies or current evaluator differ')
    if any(row.get('image') for row in jsonl(ROOT / 'data/exercism-code-v1/valid.jsonl')):
        raise ValueError('Text-only response loss comparison received image inputs')
    original = torch.load(paths[0], map_location='cpu', weights_only=True, mmap=True)
    candidate = torch.load(paths[1], map_location='cpu', weights_only=True, mmap=True)
    if (candidate['config'] != original['config'] or candidate['tokenizer_sha256'] != plan['tokenizer_sha256']
            or candidate['data_sha256'] != plan['input_manifest_sha256'] or candidate['truncated_sequences']):
        raise ValueError('Own candidate architecture/tokenizer/source allocation differs')
    with torch.device('meta'):
        parameters = TinyMultimodalLM(ModelConfig(**candidate['config'])).parameter_counts()['total']
    if parameters != plan['own_total_parameters'] or parameters >= 1_000_000_000:
        raise ValueError('Own model parameter budget differs')
    result = compare_reports(before, after)
    result.update(completed_at=datetime.now(timezone.utc).isoformat(), own_total_parameters=parameters,
        learning_steps=training['completed_steps'], actual_complete_response_targets=training['actual_response_targets'],
        full_training_report_sha256=file_hash(folder / 'training.json'), plan_sha256=file_hash(folder / 'plan.json'),
        reports_sha256={str(path.relative_to(ROOT)): file_hash(path) for label in ['before', 'after']
                        for name in FILES.values() for path in [folder / (label + '-' + name + '.json')]},
        exact_frozen_vision_and_critic_tensor_digest=training['frozen_vision_and_critic_sha256'],
        image_behavior_assessed=False, human_feedback_preference_assessed=False,
        comparison_runtime=comparison_runtime)
    if capture(source_files) != comparison_runtime:
        raise ValueError('Loaded comparison implementation changed while waiting or validating')
    atomic_json(output, result)
    atomic_json(folder / 'comparison-status.json', {'status': 'complete matched instruction comparison; broader goal unachieved',
        'comparison_sha256': file_hash(output), 'updated_at': datetime.now(timezone.utc).isoformat(), 'goal_achieved': False})
    print('Complete instruction trial comparison saved', output, flush=True)


if __name__ == '__main__':
    main()
