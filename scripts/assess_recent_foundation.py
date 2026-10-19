"""Read-only progress assessment from an open live-run periodic snapshot."""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil

import torch

from slm_lab.code_quality import evaluate as evaluate_code
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation import evaluate as evaluate_loss
from slm_lab.evaluation_provenance import capture
from slm_lab.knowledge_evaluation import evaluate as evaluate_knowledge
from slm_lab.masked_visual_pretraining import process_identity
from assess_foundation_warmup import matched

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expected-steps', type=int, required=True)
    parser.add_argument('--preceding-pid', type=int, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--device', default='cuda:2')
    args = parser.parse_args(); source = args.source.resolve(); reports = args.reports.resolve()
    if reports.exists(): raise ValueError('Preserve existing assessment; inspect its actor/artifacts before resuming')
    if process_identity(args.preceding_pid) is None or b'scripts/train_extended_curriculum.py' not in Path(f'/proc/{args.preceding_pid}/cmdline').read_bytes().split(b'\0'):
        raise ValueError('Main curriculum is not the verified live process')
    torch.set_num_threads(1)
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.2, int(args.device.split(':')[-1]))
    provenance = capture()
    controller_files = [Path(__file__), ROOT / 'scripts/assess_foundation_warmup.py', ROOT / 'scripts/train_multimodal_pilot.py']
    controller = capture(controller_files)
    fd = os.open(source, os.O_RDONLY)
    before_stat = os.fstat(fd)
    reports.mkdir(parents=True); inputs = reports / 'inference-input'; inputs.mkdir()
    alias = inputs / 'model.pt'; alias.symlink_to(f'/proc/{os.getpid()}/fd/{fd}')
    tokenizer = ROOT / 'data/tokenizer-corpus/tokenizer.json'
    (inputs / 'tokenizer.json').symlink_to(tokenizer)

    def status(message, **fields):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
                    'goal_achieved': False, **fields})

    def guard():
        if capture() != provenance or capture(controller_files) != controller:
            raise ValueError('Loaded progress-assessment implementation/runtime changed')
        now = os.fstat(fd)
        if (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns) != (before_stat.st_dev, before_stat.st_ino, before_stat.st_size, before_stat.st_mtime_ns):
            raise ValueError('Open original snapshot changed')

    def load(name):
        return json.loads((reports / name).read_text())

    try:
        status('verifying actual periodic checkpoint and target exposure', source=str(source), expected_steps=args.expected_steps)
        current = torch.load(alias, map_location='cpu', weights_only=True, mmap=True)
        stage = next(item for item in json.loads((ROOT / 'reports/extended-curriculum-v1/plan.json').read_text())['stages'] if item['name'] == 'language-pretraining')
        if (current['stage'] != 'pretrain' or current['steps'] != args.expected_steps
                or current['planned_steps'] != stage['steps'] or current['data_sha256'] != stage['data_sha256']
                or current['world_size'] != 4 or current['trainable'] != 'language' or current['truncated_sequences'] != 0):
            raise ValueError('Snapshot does not match actual live language curriculum')
        if ([item['step'] for item in current['metrics']] != list(range(1, args.expected_steps + 1))
                or any(item['global_target_tokens'] != 262080 for item in current['metrics'])):
            raise ValueError('Actual snapshot updates or target exposures incomplete')
        base = ROOT / 'checkpoints/extended-curriculum-v1/foundation-warmup/model.pt'
        old = torch.load(base, map_location='cpu', weights_only=True, mmap=True)
        if current['config'] != old['config'] or current['tokenizer_sha256'] != old['tokenizer_sha256'] or file_hash(tokenizer) != current['tokenizer_sha256']:
            raise ValueError('Own checkpoint architecture/tokenizer changed')
        vision = [name for name in old['model'] if name.startswith('vision.')]
        if not vision or any(not torch.equal(old['model'][name], current['model'][name]) for name in vision):
            raise ValueError('Language-only continuation changed frozen image tensors')
        if any(not torch.isfinite(item).all() for item in current['model'].values()): raise ValueError('Snapshot contains nonfinite model weights')
        baseline = ROOT / 'reports/foundation-warmup-assessment-v1'
        suffixes = ['code-loss', 'wikipedia-loss', 'code-completion', 'code-instruction', 'knowledge']
        base_sha = file_hash(base); source_sha = file_hash(alias)
        original = {suffix: json.loads((baseline / ('warmup-' + suffix + '.json')).read_text()) for suffix in suffixes}
        for report in original.values():
            if report['checkpoint_sha256'] != base_sha or report['evaluation_provenance'] != provenance:
                raise ValueError('Completed warmup baseline belongs to another model/evaluator')
        components = {'code': ROOT / 'data/code-search-expanded-v1', 'wikipedia': ROOT / 'data/wikipedia-foundation-warmup-v1'}
        questions = ROOT / 'data/knowledge-eval-v1/questions.jsonl'
        plan = {'source_checkpoint': str(source), 'source_checkpoint_sha256': source_sha,
                'source_steps': args.expected_steps, 'source_planned_steps': stage['steps'],
                'actual_current_stage_target_exposures': sum(item['global_target_tokens'] for item in current['metrics']),
                'warmup_checkpoint_sha256': base_sha, 'warmup_steps': old['steps'],
                'vision_tensors_preserved_exactly': len(vision), 'preceding_pid': args.preceding_pid,
                'preceding_start_ticks': process_identity(args.preceding_pid), 'device': args.device,
                'code_tasks_per_mode': 24, 'code_generation_budget': 512, 'knowledge_questions': 559,
                'loss_windows_per_component': 64, 'component_manifest_sha256': {name: file_hash(path / 'corpus.json') for name, path in components.items()},
                'knowledge_questions_sha256': file_hash(questions),
                'baseline_report_sha256': {suffix: file_hash(baseline / ('warmup-' + suffix + '.json')) for suffix in suffixes},
                'evaluation_provenance': provenance, 'controller_provenance': controller,
                'checkpoint_copy_bytes': 0, 'snapshot_kept_open_read_only_during_assessment': True,
                'goal_achieved': False, 'production_promoted': False,
                'limitations': ['Reused development protocols, not independent final or high-end comparison.',
                                'Live periodic checkpoint is held open for this assessment only and may later rotate away; no permanent inference snapshot is exported.',
                                'Exact benchmark-text screening does not exclude semantic/concept/solution overlap.',
                                'Current foundation has not yet received the queued instruction/image/joint/feedback stages.']}
        atomic_json(reports / 'plan.json', plan); del current, old; gc.collect()
        for name, folder in components.items():
            guard(); status('measuring matched held-out token loss', component=name)
            evaluate_loss(alias, folder, 'loss', reports / (name + '-loss.json'), device=args.device, threads=1, max_examples=64)
            if args.device.startswith('cuda'): torch.cuda.empty_cache()
        for mode in ['completion', 'instruction']:
            guard(); status('measuring matched generated code correctness and efficiency', mode=mode)
            evaluate_code(alias, reports / ('code-' + mode + '.json'), args.device, max_new_tokens=512, prompt_mode=mode)
            if args.device.startswith('cuda'): torch.cuda.empty_cache()
        guard(); status('measuring all reserved domain knowledge questions', examples=559)
        evaluate_knowledge(alias, questions, reports / 'knowledge.json', args.device, max_examples=559)
        guard()
        if file_hash(alias) != source_sha: raise ValueError('Open checkpoint byte hash changed during assessment')
        if any(file_hash(baseline / ('warmup-' + suffix + '.json')) != plan['baseline_report_sha256'][suffix] for suffix in suffixes):
            raise ValueError('Original warmup comparison reports changed')
        comparison = {'source_checkpoint_sha256': source_sha, 'source_steps': args.expected_steps,
                      'actual_current_stage_target_exposures': plan['actual_current_stage_target_exposures'],
                      'base_checkpoint_sha256': base_sha, 'vision_tensors_preserved_exactly': len(vision),
                      'loss': {}, 'code': {}, 'knowledge': {}, 'goal_achieved': False, 'production_promoted': False,
                      'limitations': plan['limitations']}
        for suffix in suffixes:
            result = load(suffix + '.json')
            if result['checkpoint_sha256'] != source_sha or result['evaluation_provenance'] != provenance:
                raise ValueError('Assessed report differs from current snapshot/evaluator')
        for name in components:
            first, second = original[name + '-loss'], load(name + '-loss.json')
            matched(first, second, ['data_sha256', 'sample_indices', 'target_tokens', 'examples', 'truncated_sequences'])
            if first['truncated_sequences'] or first['examples'] != 64: raise ValueError('Incomplete/truncated held loss')
            comparison['loss'][name] = {'examples': 64, 'targets': first['target_tokens'], 'before_nll': first['nll'], 'after_nll': second['nll']}
        for mode in ['completion', 'instruction']:
            first, second = original['code-' + mode], load('code-' + mode + '.json')
            matched(first, second, ['generation_budget', 'prompt_mode', 'decoding', 'suite_sha256'], ('results', ['language', 'task', 'specification']))
            if len(second['results']) != 24: raise ValueError('Incomplete generated-code assessment')
            metrics = ['task_count', 'syntax_pass_rate', 'functional_pass_at_1', 'efficiency_stress_pass_rate', 'by_language']
            comparison['code'][mode] = {'before': {key: first[key] for key in metrics}, 'after': {key: second[key] for key in metrics},
                'previously_functional_lost': [new['language'] + ':' + new['task'] for old, new in zip(first['results'], second['results'])
                                             if old['candidate'].get('functional_passed') and not new['candidate'].get('functional_passed')]}
        first, second = original['knowledge'], load('knowledge.json')
        matched(first, second, ['questions_sha256', 'contamination_audit_sha256', 'generation_budget', 'protocol'], ('predictions', ['id', 'domain', 'answer_index']))
        if first['skipped'] or second['skipped'] or second['overall']['questions'] != 559: raise ValueError('Incomplete knowledge assessment')
        comparison['knowledge'] = {'before': first['overall'], 'after': second['overall'],
                                  'by_domain': {key: {'before': first['by_domain'][key], 'after': second['by_domain'][key]} for key in first['by_domain']}}
        atomic_json(reports / 'comparison.json', comparison)
        status('recent foundation assessment completed; broad capability unverified', source_checkpoint_sha256=source_sha)
        print(json.dumps({key: value for key, value in comparison.items() if key != 'knowledge'}), flush=True)
        print(json.dumps(comparison['knowledge']['after']), flush=True)
    except Exception as error:
        status('recent foundation assessment failed; inspect without restarting live training', error=str(error))
        raise
    finally:
        alias.unlink(missing_ok=True); os.close(fd)


if __name__ == '__main__': main()
