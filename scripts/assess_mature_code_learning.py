"""Check whether additional own pretraining improves verified-program SFT learning."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.masked_visual_pretraining import process_identity

ROOT = Path(__file__).resolve().parents[1]


def compare_cases(first, second, keys, case_keys):
    for key in ['evaluation_provenance', 'tokenizer_sha256', 'inference_precision',
                'inference_device_type', *keys]:
        if first[key] != second[key]: raise ValueError('Mismatched code-learning comparison: ' + key)
    if ([[row[key] for key in case_keys] for row in first['results']]
            != [[row[key] for key in case_keys] for row in second['results']]):
        raise ValueError('Mismatched code-learning cases')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preceding-pid', required=True, type=int)
    args = parser.parse_args()
    reports = ROOT / 'reports/mature-code-learning-v1'; reports.mkdir(exist_ok=True)
    if (reports / 'plan.json').exists(): raise ValueError('Already scheduled; inspect the existing process and artifacts')
    source = ROOT / 'reports/mature-verified-code-probe-v1'
    model_root = ROOT / 'checkpoints/mature-verified-code-probe-v1'
    control = ROOT / 'checkpoints/verified-code-foundation-probe-v1/candidate/model.pt'
    control_reports = ROOT / 'reports/verified-code-foundation-probe-v1'
    data = ROOT / 'data/verified-code-curriculum-v1/train.jsonl'
    files = [ROOT / name for name in capture()['files_sha256']] + [Path(__file__),
             ROOT / 'src/slm_lab/verified_code_diagnostics.py', ROOT / 'src/slm_lab/vision_transfer.py']
    loaded = capture(files)

    def status(message, **fields):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
                                            'goal_achieved': False, **fields})

    def guard():
        if capture(files) != loaded: raise ValueError('Loaded code-learning evaluator/runtime changed')

    def run(module, checkpoint, name, extra):
        guard()
        status('assessing verified-program learning', checkpoint=str(checkpoint), assessment=name)
        subprocess.run([sys.executable, '-u', '-m', module, '--checkpoint', str(checkpoint),
                        '--output', str(reports / name), '--device', 'cuda:2', *extra],
                       cwd=ROOT, check=True, env={**os.environ, 'OMP_NUM_THREADS': '1',
                                                'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'})
        guard()

    def read(path): return json.loads(Path(path).read_text())

    try:
        identity = process_identity(args.preceding_pid)
        if identity is None: raise ValueError('Expected live mature verified-code probe')
        command = Path(f'/proc/{args.preceding_pid}/cmdline').read_bytes().split(b'\0')
        if b'scripts/train_verified_code_probe.py' not in command or b'reports/mature-verified-code-probe-v1' not in command:
            raise ValueError('Wrong code-training predecessor')
        newer, older = read(source / 'plan.json'), read(control_reports / 'plan.json')
        for key in ['parameters', 'tokenizer_sha256', 'training_manifest_sha256', 'training_data_sha256',
                    'training_rows', 'unique_source_programs', 'passes', 'targets_per_pass', 'steps',
                    'batch_size', 'gradient_accumulation', 'learning_rate', 'truncated_training_examples']:
            if newer[key] != older[key]: raise ValueError('Pretraining-depth ablation changed SFT budget/data: ' + key)
        if newer['source_completed_steps'] != 4785 or older['source_completed_steps'] != 1000:
            raise ValueError('Unexpected foundation depth')
        expected_control = read(control_reports / 'candidate-completion.json')['checkpoint_sha256']
        if file_hash(control) != expected_control or file_hash(data) != newer['training_data_sha256']:
            raise ValueError('Old control or shared training data changed')
        plan = {'preceding_pid': args.preceding_pid, 'preceding_start_ticks': identity,
                'control_checkpoint_sha256': expected_control, 'control_foundation_updates': 1000,
                'new_foundation_updates': 4785, 'source_training_plan_sha256': file_hash(source / 'plan.json'),
                'training_data_sha256': file_hash(data), 'sft_updates_both': newer['steps'],
                'seen_original_programs': 32, 'code_tasks_per_mode': 24, 'generation_budget': 512,
                'controller_evaluator_provenance': loaded, 'goal_achieved': False,
                'production_promoted': False,
                'limitations': ['Own partially trained foundations; reused development tasks, not independent final tests.',
                                'Seen examples are in original prompts and cannot establish held-out correctness.',
                                'No broad-language, domain, visual or high-end-model claim or automatic promotion.']}
        atomic_json(reports / 'plan.json', plan)
        while process_identity(args.preceding_pid) == identity:
            guard(); status('waiting for verified live mature code training/evaluation', preceding_pid=args.preceding_pid,
                            preceding_start_ticks=identity); time.sleep(15)
        guard()
        if read(source / 'status.json')['status'] != 'verified-code intervention and local before/after checks completed; broader capability unverified':
            raise ValueError('Mature code probe did not complete its declared training/evaluation')
        if file_hash(source / 'plan.json') != plan['source_training_plan_sha256']:
            raise ValueError('Mature code training plan changed')
        old_seen_path = ROOT / 'reports/verified-code-learning-diagnostic-v1/candidate.json'
        old_seen = read(old_seen_path)
        old_files = [ROOT / name for name in old_seen['evaluation_provenance']['files_sha256']]
        if (old_seen['checkpoint_sha256'] != expected_control or old_seen['data_sha256'] != file_hash(data)
                or capture(old_files) != old_seen['evaluation_provenance']):
            raise ValueError('Prior seen-example control does not match current exact model/data/evaluator')
        shutil.copy2(old_seen_path, reports / 'older-seen.json')
        for mode in ['completion', 'instruction']:
            run('slm_lab.code_quality', control, 'older-' + mode + '.json',
                ['--prompt-mode', mode, '--max-new-tokens', '512'])
        for label in ['base', 'candidate']:
            run('slm_lab.verified_code_diagnostics', model_root / label / 'model.pt', label + '-seen.json',
                ['--data', str(data), '--examples', '32', '--max-new-tokens', '512'])
        old_seen, before, after = [read(reports / (name + '-seen.json')) for name in ['older', 'base', 'candidate']]
        seen_keys = ['data_sha256', 'source_manifest_sha256', 'selection', 'examples', 'generation_budget', 'target_tokens']
        seen_cases = ['id', 'original_program_sha256', 'prompt_sha256', 'published_cases', 'target_tokens']
        compare_cases(old_seen, after, seen_keys, seen_cases); compare_cases(before, after, seen_keys, seen_cases)
        metrics = ['teacher_forced_nll', 'teacher_forced_token_accuracy', 'syntax_pass_rate', 'published_example_functional_rate']
        comparison = {'foundation_updates': [1000, 4785], 'sft_updates': newer['steps'],
                      'seen': {label: {key: report[key] for key in metrics}
                               for label, report in [('older_sft', old_seen), ('newer_base', before), ('newer_sft', after)]},
                      'held_code': {}, 'goal_achieved': False, 'production_promoted': False,
                      'limitations': plan['limitations']}
        for mode in ['completion', 'instruction']:
            first, second = read(reports / ('older-' + mode + '.json')), read(source / ('candidate-' + mode + '.json'))
            compare_cases(first, second, ['generation_budget', 'prompt_mode', 'decoding', 'suite_sha256', 'task_count'],
                          ['language', 'task', 'specification'])
            metrics = ['task_count', 'syntax_pass_rate', 'functional_pass_at_1', 'efficiency_stress_pass_rate', 'by_language']
            comparison['held_code'][mode] = {'older_sft': {key: first[key] for key in metrics},
                                            'newer_sft': {key: second[key] for key in metrics}}
        guard()
        atomic_json(reports / 'comparison.json', comparison)
        status('pretraining-depth and verified-code learning assessment completed; broader capability unverified')
        print(json.dumps(comparison, indent=2), flush=True)
    except Exception as error:
        status('requires inspection; code-learning comparison stopped', error=str(error))
        raise


if __name__ == '__main__': main()
