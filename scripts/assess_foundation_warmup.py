"""Measure the completed foundation warmup against its exact starting policy."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.masked_visual_pretraining import process_identity
from train_multimodal_pilot import complete_stage

ROOT = Path(__file__).resolve().parents[1]


def matched(before, after, keys, cases=None):
    for key in ['evaluation_provenance', 'tokenizer_sha256', 'inference_device_type',
                'inference_precision', *keys]:
        if key not in before or key not in after or before[key] != after[key]:
            raise ValueError('Unmatched foundation assessment: ' + key)
    if cases:
        field, names = cases
        first = [[row[name] for name in names] for row in before[field]]
        second = [[row[name] for name in names] for row in after[field]]
        if first != second:
            raise ValueError('Unmatched foundation assessment cases')


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preceding-pid', type=int, required=True)
    parser.add_argument('--device', default='cuda:2')
    args = parser.parse_args()
    reports = ROOT / 'reports/foundation-warmup-assessment-v1'
    reports.mkdir(exist_ok=True)
    if (reports / 'plan.json').exists():
        raise ValueError('Assessment already scheduled; inspect its process/artifacts before resuming')
    implementation = capture()
    controller = capture([Path(__file__), ROOT / 'scripts/train_multimodal_pilot.py'])

    def guard():
        if capture() != implementation or capture([Path(__file__), ROOT / 'scripts/train_multimodal_pilot.py']) != controller:
            raise ValueError('Loaded foundation assessment implementation/runtime changed')

    def status(message, **fields):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
                    'goal_achieved': False, **fields})

    def run(command):
        guard()
        subprocess.run([sys.executable, '-u', '-m', *command], check=True, cwd=ROOT,
                       env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                            'TOKENIZERS_PARALLELISM': 'false'})
        guard()

    def load(name):
        return json.loads((reports / name).read_text())

    try:
        identity = process_identity(args.preceding_pid)
        if identity is None or b'scripts/train_extended_curriculum.py' not in Path(f'/proc/{args.preceding_pid}/cmdline').read_bytes().split(b'\0'):
            raise ValueError('Expected the verified live main curriculum')
        main_reports = ROOT / 'reports/extended-curriculum-v1'
        warmup = json.loads((main_reports / 'foundation-warmup-plan.json').read_text())
        base = Path(warmup['base_checkpoint'])
        if file_hash(base) != warmup['base_checkpoint_sha256']:
            raise ValueError('Foundation starting checkpoint changed')
        checkpoint = ROOT / 'checkpoints/extended-curriculum-v1/foundation-warmup/model.pt'
        data = {'code': ROOT / 'data/code-search-expanded-v1',
                'wikipedia': ROOT / 'data/wikipedia-foundation-warmup-v1'}
        questions = ROOT / 'data/knowledge-eval-v1/questions.jsonl'
        question_hash = file_hash(questions)
        for source in ['code-search-expanded-v1', 'wikipedia-expanded-v1']:
            config = json.loads((ROOT / 'data' / source / 'collection-config.json').read_text())
            if config['questions_sha256'] != question_hash:
                raise ValueError('Expanded source did not protect these exact reserved questions')
        plan = {'base_checkpoint': str(base), 'base_checkpoint_sha256': file_hash(base),
                'warmup_plan_sha256': file_hash(main_reports / 'foundation-warmup-plan.json'),
                'expected_updates': warmup['steps'], 'target_token_exposures': warmup['planned_target_tokens'],
                'preceding_pid': args.preceding_pid, 'preceding_start_ticks': identity,
                'device': args.device, 'loss_windows_per_component': 64,
                'component_manifest_sha256': {name: file_hash(path / 'corpus.json') for name, path in data.items()},
                'knowledge_questions_sha256': question_hash, 'knowledge_examples': 559,
                'code_tasks_per_prompt_mode': 24, 'code_generation_budget': 512,
                'evaluation_provenance': implementation, 'controller_provenance': controller,
                'goal_achieved': False, 'production_promoted': False,
                'limitations': ['Repeated development tasks; no independent final assessment or high-end comparison.',
                                'Reserved exact knowledge questions were filtered at collection; semantic overlap remains possible.',
                                'Token loss and multiple-choice likelihood do not prove useful code, factual reliability or reasoning.']}
        atomic_json(reports / 'plan.json', plan)

        def assess(path, label):
            expected = file_hash(path)
            for component, folder in data.items():
                status('measuring held-out foundation token loss', checkpoint=label, component=component)
                run(['slm_lab.evaluation', '--checkpoint', str(path), '--data', str(folder), '--task', 'loss',
                     '--max-examples', '64', '--threads', '1', '--device', args.device,
                     '--output', str(reports / f'{label}-{component}-loss.json')])
            for mode in ['completion', 'instruction']:
                status('measuring generated code correctness and efficiency', checkpoint=label, prompt_mode=mode)
                run(['slm_lab.code_quality', '--checkpoint', str(path), '--prompt-mode', mode,
                     '--device', args.device, '--output', str(reports / f'{label}-code-{mode}.json')])
            status('measuring all reserved domain knowledge questions', checkpoint=label)
            run(['slm_lab.knowledge_evaluation', '--checkpoint', str(path), '--data', str(questions),
                 '--max-examples', '559', '--device', args.device,
                 '--output', str(reports / f'{label}-knowledge.json')])
            if file_hash(path) != expected:
                raise ValueError('Assessed checkpoint changed')
            for suffix in ['code-loss', 'wikipedia-loss', 'code-completion', 'code-instruction', 'knowledge']:
                report = load(f'{label}-{suffix}.json')
                if report['checkpoint_sha256'] != expected or report['evaluation_provenance'] != implementation:
                    raise ValueError('Report differs from exact assessed model/implementation')
            return expected

        assess(base, 'base')
        completed_path = main_reports / 'foundation-warmup-completed.json'
        while not completed_path.exists():
            if process_identity(args.preceding_pid) != identity:
                raise ValueError('Main process ended without completed foundation warmup; inspect training')
            guard()
            status('waiting for verified live foundation warmup to finish', preceding_pid=args.preceding_pid,
                   preceding_start_ticks=identity, expected_updates=warmup['steps'])
            time.sleep(15)
        completed = json.loads(completed_path.read_text())
        if (completed['plan_sha256'] != plan['warmup_plan_sha256']
                or completed['completed_steps'] != warmup['steps']
                or completed['checkpoint_sha256'] != file_hash(checkpoint)
                or not complete_stage(checkpoint.parent, warmup)):
            raise ValueError('Completed foundation warmup does not match its exact plan/weights')
        before = torch.load(base, weights_only=True, map_location='cpu', mmap=True)
        after = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
        if before['config'] != after['config'] or before['tokenizer_sha256'] != after['tokenizer_sha256']:
            raise ValueError('Foundation architecture/tokenizer changed')
        vision_names = [name for name in before['model'] if name.startswith('vision.')]
        if not vision_names or any(not torch.equal(before['model'][name], after['model'][name]) for name in vision_names):
            raise ValueError('Language-only foundation warmup changed vision weights')
        if any(not torch.isfinite(value).all() for value in after['model'].values()):
            raise ValueError('Completed foundation weights are nonfinite')
        del before, after
        candidate_sha = assess(checkpoint, 'warmup')
        comparison = {'completed_updates': warmup['steps'], 'target_token_exposures': warmup['planned_target_tokens'],
                      'base_checkpoint_sha256': plan['base_checkpoint_sha256'], 'candidate_checkpoint_sha256': candidate_sha,
                      'vision_tensors_preserved_exactly': len(vision_names), 'goal_achieved': False,
                      'production_promoted': False, 'loss': {}, 'code': {}, 'knowledge': {}}
        for component in data:
            first, second = load(f'base-{component}-loss.json'), load(f'warmup-{component}-loss.json')
            matched(first, second, ['data_sha256', 'sample_indices', 'target_tokens', 'examples', 'truncated_sequences'])
            if first['truncated_sequences'] or first['examples'] != 64:
                raise ValueError('Incomplete/truncated foundation loss assessment')
            comparison['loss'][component] = {'examples': first['examples'], 'targets': first['target_tokens'],
                'before_nll': first['nll'], 'after_nll': second['nll'],
                'before_perplexity': first['perplexity'], 'after_perplexity': second['perplexity']}
        for mode in ['completion', 'instruction']:
            first, second = load(f'base-code-{mode}.json'), load(f'warmup-code-{mode}.json')
            matched(first, second, ['generation_budget', 'prompt_mode', 'decoding', 'suite_sha256'],
                    ('results', ['language', 'task', 'specification']))
            if len(first['results']) != 24:
                raise ValueError('Incomplete generated-code assessment')
            metrics = ['task_count', 'syntax_pass_rate', 'functional_pass_at_1', 'efficiency_stress_pass_rate', 'by_language']
            comparison['code'][mode] = {'before': {name: first[name] for name in metrics},
                'after': {name: second[name] for name in metrics},
                'previously_functional_lost': [row['language'] + ':' + row['task']
                    for old, row in zip(first['results'], second['results'])
                    if old['candidate'].get('functional_passed') and not row['candidate'].get('functional_passed')]}
        first, second = load('base-knowledge.json'), load('warmup-knowledge.json')
        matched(first, second, ['questions_sha256', 'contamination_audit_sha256', 'generation_budget', 'protocol'],
                ('predictions', ['id', 'domain', 'answer_index']))
        if first['skipped'] or second['skipped'] or first['overall']['questions'] != 559:
            raise ValueError('Incomplete reserved knowledge assessment')
        comparison['knowledge'] = {'before': first['overall'], 'after': second['overall'],
                'by_domain': {name: {'before': first['by_domain'][name], 'after': second['by_domain'][name]}
                              for name in first['by_domain']}}
        comparison['limitations'] = plan['limitations']
        atomic_json(reports / 'comparison.json', comparison)
        status('foundation warmup assessment completed; broader capability unverified', checkpoint_sha256=candidate_sha)
        print(json.dumps(comparison, indent=2), flush=True)
    except Exception as error:
        status('requires inspection; foundation assessment stopped', error=str(error))
        raise


if __name__ == '__main__':
    main()
