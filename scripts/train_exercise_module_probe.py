"""Test six-language complete-module SFT on own completed coding weights."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
from tokenizers import Tokenizer

from slm_lab.completed_state_cleanup import retire
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation_provenance import capture
from slm_lab.exercise_cpp_selection import compare_reports as compare_cpp
from slm_lab.exercise_evaluation import compare_reports as compare_python
from slm_lab.exercise_probe import exposure_plan, validate_splits
from slm_lab.masked_visual_pretraining import process_identity
from slm_lab.model import ModelConfig, TinyMultimodalLM
from slm_lab.training import Encoder
from train_multimodal_pilot import complete_stage

ROOT = Path(__file__).resolve().parents[1]
PRECEDING_DONE = 'matched native supervised-vision experiment completed; broader capability unverified'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preceding-pid', required=True, type=int)
    args = parser.parse_args(); torch.set_num_threads(1)
    reports = ROOT / 'reports/exercism-module-probe-v1'
    checkpoints = ROOT / 'checkpoints/exercism-module-probe-v1'
    if reports.exists() or checkpoints.exists(): raise ValueError('Existing module probe requires explicit inspection; do not relaunch')
    reports.mkdir()
    data = ROOT / 'data/exercism-code-v1'
    source = ROOT / 'checkpoints/mature-verified-code-probe-v1/candidate/model.pt'
    files = [ROOT / name for name in capture()['files_sha256']] + [Path(__file__)]
    files += [ROOT / 'src/slm_lab' / (name + '.py') for name in
              ['exercise_probe', 'exercise_code', 'exercise_evaluation', 'exercise_cpp_evaluation',
               'exercise_cpp_runner', 'exercise_cpp_selection', 'completed_state_cleanup', 'masked_visual_pretraining']]
    loaded = capture(files)

    def status(message, **fields):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
                    'goal_achieved': False, 'production_promoted': False, **fields})

    def guard(full=True):
        if capture(files) != loaded: raise ValueError('Module-probe source/runtime changed; inspect before continuing')
        for path, digest in plan['inputs_sha256'].items():
            if not full and path.endswith('.pt'):
                stat = (ROOT / path).stat()
                if [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns] != plan['source_file_identity']:
                    raise ValueError('Immutable source model file identity changed while waiting')
                continue
            if file_hash(ROOT / path) != digest: raise ValueError('Frozen module-probe input changed: ' + path)

    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '2', 'OMP_NUM_THREADS': '1',
                   'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}

    def run(command):
        guard(); subprocess.run(command, cwd=ROOT, check=True, env=environment); guard()

    def assess(module, checkpoint, name, extra=None, reuse=None):
        output = reports / name
        if reuse is not None:
            old = json.loads(Path(reuse).read_text())
            old_files = [ROOT / name for name in old['evaluation_provenance']['files_sha256']]
            if (old['checkpoint_sha256'] != file_hash(checkpoint) or old['tokenizer_sha256'] != plan['tokenizer_sha256'] or
                old['inference_precision'] != 'bf16' or old['inference_device_type'] != 'cuda' or
                capture(old_files) != old['evaluation_provenance']):
                raise ValueError('Cached baseline does not match exact weights/tokenizer/current evaluator')
            shutil.copy2(reuse, output); return
        status('assessing complete-module coding experiment', checkpoint=str(checkpoint), assessment=name)
        run([sys.executable, '-u', '-m', module, '--checkpoint', str(checkpoint), '--output', str(output),
             '--device', 'cuda', *(extra or [])])

    try:
        preceding = args.preceding_pid; identity = process_identity(preceding)
        if identity is None or b'scripts/train_supervised_vision_alignment.py' not in Path(f'/proc/{preceding}/cmdline').read_bytes().split(b'\0'):
            raise ValueError('Expected the currently live own native-vision controller before queuing GPU2 work')
        manifest = json.loads((data / 'manifest.json').read_text())
        audit = json.loads((ROOT / 'reports/exercism-input-audit-v1.json').read_text())
        if audit['manifest_sha256'] != file_hash(data / 'manifest.json') or not audit['original_archive_source_and_test_artifact_hashes_verified']:
            raise ValueError('Exercise source audit differs from the frozen manifest')
        for name, digest in manifest['files_sha256'].items():
            if file_hash(data / name) != digest: raise ValueError('Exercise split changed')
        train, valid = list(jsonl(data / 'train.jsonl')), list(jsonl(data / 'valid.jsonl'))
        validate_splits(train, valid)
        if len(train) != 599 or len(valid) != 181: raise ValueError('Unexpected source-grounded exercise corpus')
        initial = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
        completion = json.loads((source.parent / 'training.json').read_text())
        expected_source = json.loads((ROOT / 'reports/exercism-mature-cpp-baseline-v1.json').read_text())['checkpoint_sha256']
        if (file_hash(source) != expected_source or initial['stage'] != 'sft' or initial['steps'] != completion['planned_steps'] or
            completion['steps'] != completion['planned_steps'] or 'optimizer' in initial or 'reference_model' in initial):
            raise ValueError('Expected exact own completed weights-only coding checkpoint')
        config = ModelConfig(**initial['config'])
        tokenizer = source.parent / 'tokenizer.json'
        if file_hash(tokenizer) != initial['tokenizer_sha256'] or file_hash(tokenizer) != manifest['tokenizer_sha256']:
            raise ValueError('Own model/exercise tokenizer changed')
        encoder = Encoder(Tokenizer.from_file(str(tokenizer)), config, data)
        counts = []; maximum = 0
        for row in train:
            ids, labels = encoder.sequence(row, 'response'); count = sum(value != -100 for value in labels[1:])
            if len(ids) != row['context_tokens'] or count != row['target_tokens']:
                raise ValueError('Original module token accounting changed')
            counts.append(count); maximum = max(maximum, len(ids))
        if encoder.truncated: raise ValueError('Complete-module SFT would truncate a source target')
        if any(not torch.isfinite(value).all() for value in initial['model'].values()): raise ValueError('Invalid source model weights')
        with torch.device('meta'): parameters = TinyMultimodalLM(config).parameter_counts()['total']
        if parameters >= 1_000_000_000: raise ValueError('Model exceeds the user parameter limit')
        budget, exposures = exposure_plan(train, counts)
        base = checkpoints / 'base/model.pt'; base.parent.mkdir(parents=True)
        os.link(source, base); shutil.copy2(tokenizer, base.parent / 'tokenizer.json')
        source_stat = source.stat()
        with (reports / 'exposures.jsonl').open('w') as out:
            for record in exposures: out.write(json.dumps(record) + '\n')
        input_paths = [source, tokenizer, source.parent / 'training.json', data / 'manifest.json',
                       data / 'train.jsonl', data / 'valid.jsonl', ROOT / 'reports/exercism-input-audit-v1.json',
                       ROOT / 'reports/exercism-cpp-reference-readiness-v1.json']
        plan = {**budget, 'parameters': parameters, 'source_checkpoint': str(source), 'base_checkpoint': str(base),
                'source_checkpoint_sha256': expected_source, 'tokenizer_sha256': manifest['tokenizer_sha256'],
                'training_data_sha256': manifest['files_sha256']['train.jsonl'], 'maximum_sequence_tokens': maximum,
                'training_languages': manifest['splits']['train']['languages'], 'validation_languages': manifest['splits']['valid']['languages'],
                'validation_exercise_groups': manifest['splits']['valid']['exercise_groups'], 'learning_rate': 1e-5,
                'trainable': 'language', 'vision_frozen': True, 'truncated_training_examples': 0,
                'preceding_pid': preceding, 'preceding_start_ticks': identity, 'physical_gpu': 2,
                'source_file_identity': [source_stat.st_dev, source_stat.st_ino, source_stat.st_size,
                                         source_stat.st_mtime_ns, source_stat.st_ctime_ns],
                'periodic_resume_snapshots': 0,
                'implementations': loaded, 'inputs_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in input_paths},
                'exposure_ledger_sha256': file_hash(reports / 'exposures.jsonl'),
                'scope': 'Separate six-language original complete-module SFT and matched code/loss assessments; no production promotion.',
                'goal_achieved': False, 'production_promoted': False,
                'limitations': manifest['limitations'] + ['Execution covers supported Python literals and original C++ publisher suites; other languages assessed by response loss only.',
                               'Reused development tasks, no independent final generalization or high-end capability claim.',
                               'Full knowledge/math/dialogue/visual/preference regressions are not established by this separate probe.']}
        atomic_json(reports / 'plan.json', plan); del initial
        while process_identity(preceding) == identity:
            guard(full=False); status('waiting for verified live native-vision experiment to release GPU2',
                            preceding_pid=preceding, preceding_start_ticks=identity, planned_steps=budget['steps']); time.sleep(15)
        guard()
        if json.loads((ROOT / 'reports/supervised-vision-alignment-v1/status.json').read_text())['status'] != PRECEDING_DONE:
            raise ValueError('Native-vision predecessor did not finish its declared work')
        if shutil.disk_usage(ROOT).free < 35 * 1024**3: raise ValueError('Need35 GiB free before separate module-probe training')
        # Reuse only reports whose exact immutable source/model/evaluator was verified above.
        refs = ROOT / 'reports/exercism-cpp-reference-readiness-v1.json'
        exercise_args = ['--data', str(data / 'valid.jsonl')]
        assessments = [('slm_lab.evaluation', 'loss', exercise_args + ['--task', 'loss', '--max-examples', '181', '--threads', '1']),
                       ('slm_lab.exercise_evaluation', 'python', exercise_args),
                       ('slm_lab.exercise_cpp_evaluation', 'cpp', exercise_args + ['--reference-report', str(refs)]),
                       ('slm_lab.code_quality', 'completion', ['--prompt-mode', 'completion', '--max-new-tokens', '512']),
                       ('slm_lab.code_quality', 'instruction', ['--prompt-mode', 'instruction', '--max-new-tokens', '512'])]
        reuse = {'cpp': ROOT / 'reports/exercism-mature-cpp-baseline-v1.json',
                 'completion': ROOT / 'reports/mature-verified-code-probe-v1/candidate-completion.json',
                 'instruction': ROOT / 'reports/mature-verified-code-probe-v1/candidate-instruction.json'}
        for module, name, extra in assessments: assess(module, base, 'before-' + name + '.json', extra, reuse.get(name))
        candidate = checkpoints / 'candidate'
        spec = {'stage': 'sft', 'steps': budget['steps'], 'trainable': 'language', 'image_size': 448,
                'data_sha256': manifest['files_sha256']['train.jsonl']}
        status('training six-language original complete modules', steps=budget['steps'], planned_target_exposures=budget['planned_target_exposures'])
        run([sys.executable, '-u', '-m', 'slm_lab.training', 'sft', '--checkpoint', str(base), '--tokenizer', str(tokenizer),
             '--data', str(data / 'train.jsonl'), '--output', str(candidate), '--device', 'cuda', '--trainable', 'language',
             '--steps', str(budget['steps']), '--batch-size', '2', '--gradient-accumulation', '4', '--learning-rate', '1e-5',
             '--seed', '42', '--image-size', '448', '--threads', '1', '--gradient-checkpointing', '--checkpoint-every', '0',
             '--keep-checkpoints', '1', '--log-every', '10'])
        if not complete_stage(candidate, spec): raise ValueError('Module-training budget did not complete')
        result = json.loads((candidate / 'training.json').read_text())
        if result['truncated_sequences'] or sum(m['global_target_tokens'] for m in result['metrics']) != budget['planned_target_exposures']:
            raise ValueError('Actual training exposures/truncation differ from complete-module plan')
        for path in candidate.glob('step-*.pt'): path.unlink()
        before = torch.load(base, map_location='cpu', weights_only=True, mmap=True)
        after = torch.load(candidate / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
        vision = [name for name in before['model'] if name.startswith('vision.')]
        if before['config'] != after['config'] or any(not torch.equal(before['model'][name], after['model'][name]) for name in vision):
            raise ValueError('Module SFT changed architecture or frozen vision weights')
        atomic_json(reports / 'vision-preservation.json', {'exact_frozen_vision_tensors': len(vision),
                    'base_checkpoint_sha256': expected_source, 'candidate_checkpoint_sha256': file_hash(candidate / 'model.pt'), 'goal_achieved': False})
        del before, after
        for module, name, extra in assessments: assess(module, candidate / 'model.pt', 'after-' + name + '.json', extra)
        read = lambda name: json.loads((reports / name).read_text())
        comparison = {'python_guard': compare_python(read('before-python.json'), read('after-python.json')),
                      'cpp_guard': compare_cpp(read('before-cpp.json'), read('after-cpp.json')),
                      'base_checkpoint_sha256': expected_source, 'candidate_checkpoint_sha256': file_hash(candidate / 'model.pt'),
                      'goal_achieved': False, 'production_promoted': False, 'limitations': plan['limitations'], 'local_functions': {}}
        for name in ['completion', 'instruction']:
            first, second = read('before-' + name + '.json'), read('after-' + name + '.json')
            for key in ['evaluation_provenance', 'tokenizer_sha256', 'inference_precision', 'inference_device_type',
                        'generation_budget', 'decoding', 'prompt_mode', 'suite_sha256', 'task_count']:
                if first[key] != second[key]: raise ValueError('Local code evaluation mismatch: ' + key)
            if ([(r['language'], r['task'], r['specification']) for r in first['results']] !=
                [(r['language'], r['task'], r['specification']) for r in second['results']]):
                raise ValueError('Local code generation cases differ')
            comparison['local_functions'][name] = {key: {'before': first[key], 'after': second[key]} for key in
                         ['syntax_pass_rate', 'functional_pass_at_1', 'efficiency_stress_pass_rate', 'by_language']}
        loss_before, loss_after = read('before-loss.json'), read('after-loss.json')
        for key in ['evaluation_provenance', 'tokenizer_sha256', 'data_sha256', 'task', 'examples',
                    'available_examples', 'sample_indices', 'target_tokens', 'truncated_sequences',
                    'inference_precision', 'inference_device_type']:
            if loss_before[key] != loss_after[key]: raise ValueError('Unmatched whole validation-module response loss')
        if loss_after['truncated_sequences']: raise ValueError('Validation-module assessment truncated source targets')
        groups = lambda report: {key: (value['target_tokens'], value['examples']) for key, value in report['by_group'].items()}
        if groups(loss_before) != groups(loss_after): raise ValueError('Validation-module language groups differ')
        comparison['validation_response_loss'] = {'before': loss_before, 'after': loss_after}
        guard(); atomic_json(reports / 'comparison.json', comparison)
        retire(candidate, reports / 'completed-state-cleanup.json')
        status('six-language module SFT and matched coding assessments completed; broader capability unverified')
    except Exception as error:
        status('requires inspection; complete-module coding probe stopped', error=str(error)); raise


if __name__ == '__main__': main()
