"""A single own-model SFT intervention with broader code and explicit rehearsal."""
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import torch
from tokenizers import Tokenizer

from slm_lab.binding_retention_inputs_v1 import update_rows
from slm_lab.checkpoint_xor import archive_checkpoint, restore
from slm_lab.code_quality import evaluate as code_evaluate
from slm_lab.corpus import file_hash
from slm_lab.cpp_evaluation import evaluate as cpp_evaluate
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation import evaluate as loss_evaluate
from slm_lab.evaluation_provenance import capture
from slm_lab.instruction_generalization_trial_v3 import LANGUAGES
from slm_lab.instruction_trial_comparison import compare_reports, same, COMMON
from slm_lab.knowledge_evaluation import evaluate as knowledge_evaluate
from slm_lab.model import ModelConfig, TinyMultimodalLM
from slm_lab.recurrent_ablation import frozen_digest, language_parameters, state_digest
from slm_lab.training import Encoder, supervised_gradients
from slm_lab.verified_code_diagnostics_v2 import evaluate as seen_evaluate

ROOT = Path(__file__).resolve().parents[1]
STEPS = 1500
SEED = 88125
FAMILIES = {'code': 'code-completion', 'code_instruction': 'code-instruction', 'cpp': 'cpp',
            'module_loss': 'module-loss', 'knowledge': 'knowledge'}


def main():
    torch.set_num_threads(1); torch.manual_seed(SEED)
    reports = ROOT / 'reports/binding-retention-v1'; durable = ROOT / 'checkpoints/binding-retention-v1'
    if reports.exists() or durable.exists(): raise ValueError('Preserve earlier intervention; never relaunch automatically')
    reports.mkdir(); durable.mkdir()
    source = ROOT / 'checkpoints/instruction-generalization-probe-v3/data'
    supplement = ROOT / 'data/parameter-binding-curriculum-v1'; retention = ROOT / 'data/seen-code-retention-v1'
    ready_path = ROOT / 'reports/parameter-binding-curriculum-v1/schedule-readiness.json'
    base = ROOT / 'checkpoints/instruction-generalization-probe-v3/candidate/model.pt'
    token = base.parent / 'tokenizer.json'; valid = ROOT / 'data/exercism-code-v1/valid.jsonl'
    questions = ROOT / 'data/knowledge-eval-v1/questions.jsonl'
    manifests = {p: json.loads((p / 'manifest.json').read_text()) for p in [source, supplement, retention]}
    ready = json.loads(ready_path.read_text())
    files = list(dict.fromkeys([ROOT / name for name in capture()['files_sha256']] + [Path(__file__)] +
        [ROOT / 'src/slm_lab' / (n + '.py') for n in ['binding_retention_inputs_v1',
          'instruction_generalization_trial_v3', 'instruction_trial_comparison', 'alignment_selection',
          'recurrent_ablation', 'checkpoint_xor', 'checkpoint_archive', 'verified_code_diagnostics_v2', 'vision_transfer']]))
    runtime = capture(files)
    bound = {p: file_hash(p) for p in [base, token, ready_path, valid, questions]}
    for directory, manifest in manifests.items():
        bound[directory / 'manifest.json'] = file_hash(directory / 'manifest.json')
        for name, digest in manifest['files_sha256'].items():
            p = directory / name
            if file_hash(p) != digest: raise ValueError('Prepared public source differs')
            bound[p] = digest
    for name, digest in ready['bound_files_sha256'].items():
        p = ROOT / name
        if file_hash(p) != digest: raise ValueError('Verified future exposure source differs')
        bound[p] = digest
    exposure_path = ROOT / 'reports/parameter-binding-curriculum-v1/proposed-exposures.jsonl'
    if file_hash(exposure_path) != ready['proposed_exposures_sha256']: raise ValueError('Verified exposure plan differs')
    bound[exposure_path] = file_hash(exposure_path)
    def guard():
        if capture(files) != runtime: raise ValueError('Loaded intervention source/runtime changed')
        for path, digest in bound.items():
            if file_hash(path) != digest: raise ValueError('Bound source changed: ' + str(path))
    def status(message, **fields):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
            'automatic_policy_promotion': False, 'goal_achieved': False, **fields})
    def clean(): gc.collect(); torch.cuda.empty_cache()
    location = None
    try:
        prior = ROOT / 'reports/semantic-code-optimization-v1'
        if json.loads((prior / 'status.json').read_text())['status'] != 'controlled semantic optimization and matched assessments completed; broader capability unverified':
            raise ValueError('Inspect preceding controlled experiment before further learning')
        metadata = torch.load(base, map_location='cpu', weights_only=True, mmap=True)
        config = ModelConfig(**metadata['config']); base_state = state_digest(metadata['model'])
        if metadata['steps'] != 1500 or metadata['planned_steps'] != 1500 or metadata['tokenizer_sha256'] != bound[token]:
            raise ValueError('Exact own instruction base/tokenizer differs')
        with torch.device('meta'): parameters = TinyMultimodalLM(config).parameter_counts()['total']
        if parameters != 706012672 or parameters >= 1_000_000_000: raise ValueError('Own parameter budget differs')
        del metadata
        encoder = Encoder(Tokenizer.from_file(str(token)), config, source, 448)
        pools = {name: list(jsonl(source / (name + '.jsonl'))) for name in [*LANGUAGES, 'verified_python', 'human_instruction']}
        pools['verified_python'] += list(jsonl(supplement / 'train.jsonl')); pools['retention'] = list(jsonl(retention / 'train.jsonl'))
        primary = list(jsonl(exposure_path)); rehearsal = list(jsonl(retention / 'proposed-exposures.jsonl'))
        if len(primary) != 4 * STEPS or len(rehearsal) != STEPS: raise ValueError('Exact verified exposure budget differs')
        allocation = []; target_total = 0
        for step in range(STEPS):
            picks = primary[4 * step:4 * step + 4]; replay = rehearsal[step]
            if replay['update'] != step + 1: raise ValueError('Scheduled update differs')
            rows = update_rows(pools, picks, replay, encoder); allocation.append(rows)
            target_total += sum(r['target_tokens'] for r in [*picks, replay])
        if target_total != ready['actual_complete_response_targets'] + manifests[retention]['proposed_additional_response_targets']:
            raise ValueError('Complete mixed/rehearsal target totals differ')
        baseline = {}
        for family, suffix in FAMILIES.items():
            p = ROOT / 'reports/instruction-generalization-probe-v3' / ('after-' + suffix + '.json')
            d = json.loads(p.read_text())
            if d['checkpoint_sha256'] != bound[base] or d['evaluation_provenance'] != capture(): raise ValueError('Exact own-base evaluation differs')
            baseline[family] = d; bound[p] = file_hash(p); shutil.copy2(p, reports / ('before-' + suffix + '.json'))
        compare_reports(baseline, baseline)
        if baseline['module_loss']['data_sha256'] != bound[valid] or baseline['knowledge']['questions_sha256'] != bound[questions]:
            raise ValueError('Full assessment source differs')
        before_seen_path = ROOT / 'reports/exposed-code-learning-after-instruction-v1/after.json'
        before_seen = json.loads(before_seen_path.read_text()); bound[before_seen_path] = file_hash(before_seen_path)
        if before_seen['checkpoint_sha256'] != bound[base] or before_seen['examples'] != 128:
            raise ValueError('Original source-learning baseline differs')
        same(before_seen, baseline['code'], ['inference_device_type', 'inference_precision', 'tokenizer_sha256'])
        if capture([ROOT / name for name in before_seen['evaluation_provenance']['files_sha256']]) != before_seen['evaluation_provenance']:
            raise ValueError('Original source diagnostic runtime differs')
        plan = {'recorded_at': datetime.now(timezone.utc).isoformat(), 'source_runtime': runtime,
            'base_checkpoint': str(base), 'base_checkpoint_sha256': bound[base], 'base_model_state_sha256': base_state,
            'own_total_parameters': parameters, 'steps': STEPS, 'seed': SEED, 'device': 'cuda:3',
            'actual_complete_response_targets_planned': target_total, 'binding_exposures': ready['selected_binding_exposures'],
            'original_algorithms': ready['source_algorithms'], 'retention_programs': 128,
            'objective': 'Global target-token mean NLL over four mixed examples, plus0.25 original-public-program rehearsal NLL.',
            'learning_rate': 2e-5, 'warmup_updates': 150, 'single_before_after_intervention': True,
            'isolated_argument_augmentation_causal_effect_claimed': False, 'human_feedback': False,
            'external_teacher_used': False, 'persistent_reserve_GiB': 26,
            'frozen': ['vision', 'critic'], 'automatic_policy_promotion': False, 'goal_achieved': False}
        atomic_json(reports / 'plan.json', plan); bound[reports / 'plan.json'] = file_hash(reports / 'plan.json')
        required = base.stat().st_size + 26 * 1024**3 + 16 * 1024**2
        while shutil.disk_usage(ROOT).free < required:
            guard(); status('waiting for persistent checkpoint capacity; training not started', required_bytes=required,
                            free_bytes=shutil.disk_usage(ROOT).free); time.sleep(15)
        if shutil.disk_usage('/tmp').free < 12 * 1024**3: raise ValueError('Require temporary full-checkpoint/archive capacity')
        location = Path(tempfile.mkdtemp(prefix='slm-binding-retention-v1-', dir='/tmp'))
        torch.cuda.set_per_process_memory_fraction(.5, 3)
        metadata = torch.load(base, map_location='cpu', weights_only=True, mmap=True)
        model = TinyMultimodalLM(config); model.load_state_dict(metadata['model']); del metadata
        if state_digest(model.state_dict()) != base_state: raise ValueError('Actual initialization differs')
        frozen = frozen_digest(model); trainable = language_parameters(model, False)
        model.to('cuda:3'); model.gradient_checkpointing = True; model.train()
        optimizer = torch.optim.AdamW(trainable, lr=2e-5, weight_decay=.01, foreach=False)
        amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
        metrics = []; targets = 0; started = time.monotonic(); torch.cuda.reset_peak_memory_stats(3)
        for step, rows in enumerate(allocation):
            optimizer.zero_grad(set_to_none=True)
            lr = 2e-5 * min(1, (step + 1) / 150) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / STEPS)))
            for parameter_group in optimizer.param_groups: parameter_group['lr'] = lr
            tick = time.monotonic()
            loss, count = supervised_gradients(model, encoder, [[r] for r in rows[:4]], 'response', 'cuda:3', amp)
            batch = encoder.batch([rows[4]], 'response', 'cuda:3')
            with amp(): replay_loss = model(**batch)['loss']
            if not torch.isfinite(replay_loss): raise ValueError('Nonfinite original-code rehearsal loss')
            (.25 * replay_loss).backward()
            replay_count = int((batch['labels'][:, 1:] != -100).sum()); targets += count + replay_count
            norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1, error_if_nonfinite=True)); optimizer.step(); torch.cuda.synchronize(3)
            metrics.append({'step': step + 1, 'mixed_nll': float(loss), 'rehearsal_nll': float(replay_loss.detach()),
                'objective': float(loss) + .25 * float(replay_loss.detach()), 'actual_target_tokens': count + replay_count,
                'learning_rate': lr, 'gradient_norm_before_clipping': norm, 'step_seconds': time.monotonic() - tick})
            del loss, replay_loss, batch
            if step == 0 or (step + 1) % 10 == 0:
                status('training broader public code with argument binding and original-program rehearsal',
                       completed_steps=step + 1, planned_steps=STEPS, actual_response_targets=targets)
                print(json.dumps(metrics[-1]), flush=True)
            if (step + 1) % 100 == 0: guard()
        if targets != target_total or encoder.truncated or frozen_digest(model) != frozen:
            raise ValueError('Actual target budget, complete masks or frozen tensors differ')
        optimizer.zero_grad(set_to_none=True)
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}; state_sha = state_digest(state)
        candidate = location / 'model.pt'
        torch.save({'config': dict(config.__dict__), 'model': state, 'stage': 'sft', 'steps': STEPS, 'planned_steps': STEPS,
            'tokenizer_sha256': bound[token], 'data_sha256': bound[reports / 'plan.json'], 'trainable': 'language',
            'truncated_sequences': 0, 'model_state_sha256': state_sha, 'source_base_checkpoint_sha256': bound[base]}, candidate)
        shutil.copy2(token, location / 'tokenizer.json')
        training = {'status': 'broader binding/rehearsal training completed; capability unverified', 'steps': STEPS,
            'planned_steps': STEPS, 'actual_response_targets': targets, 'metrics': metrics,
            'base_checkpoint_sha256': bound[base], 'candidate_checkpoint_sha256': file_hash(candidate),
            'candidate_model_state_sha256': state_sha, 'frozen_vision_and_critic_sha256': frozen,
            'elapsed_seconds': time.monotonic() - started, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(3), 'goal_achieved': False}
        atomic_json(reports / 'training.json', training); atomic_json(location / 'training.json', training)
        del state, model, trainable, optimizer, parameter_group; clean()
        staged = location / 'model.pt.xor'; archive = durable / 'model.pt.xor'
        archived = archive_checkpoint(candidate, base, staged, remove_original=False)
        while shutil.disk_usage(ROOT).free < archived['payload_bytes'] + 26 * 1024**3:
            guard(); status('waiting to preserve completed policy; temporary full checkpoint retained', temporary=str(candidate)); time.sleep(15)
        copying = archive.with_name(archive.name + '.staging')
        shutil.copytree(staged, copying)
        for item in copying.iterdir():
            with item.open('rb') as stream: os.fsync(stream.fileno())
        restore(copying); copying.rename(archive)
        atomic_json(reports / 'durable-archive.json', {'archive': str(archive), 'temporary_full_checkpoint': str(candidate),
            'mandatory_full_anchor': str(base), 'whole_file_restored_exact': True, 'goal_achieved': False})
        for mode in ['completion', 'instruction']:
            status('evaluating completed binding/rehearsal policy', family='code', mode=mode)
            code_evaluate(candidate, reports / ('after-code-' + mode + '.json'), 'cuda:3', max_new_tokens=512, prompt_mode=mode); clean(); guard()
        status('evaluating completed binding/rehearsal policy', family='native C++')
        cpp_evaluate(candidate, reports / 'after-cpp.json', 'cuda:3', max_new_tokens=512); clean(); guard()
        status('evaluating completed binding/rehearsal policy', family='full181 module loss')
        loss_evaluate(candidate, valid, 'loss', reports / 'after-module-loss.json', device='cuda:3', threads=1, max_examples=181); clean(); guard()
        status('evaluating completed binding/rehearsal policy', family='full559 knowledge')
        knowledge_evaluate(candidate, questions, reports / 'after-knowledge.json', 'cuda:3', max_examples=559); clean(); guard()
        status('evaluating completed binding/rehearsal policy', family='original128 exposed programs')
        seen_evaluate(candidate, source / 'verified_python.jsonl', reports / 'after-seen.json', 'cuda:3',
                      count=128, max_new_tokens=512, exposures=source / 'exposures.jsonl'); clean(); guard()
        after = {family: json.loads((reports / ('after-' + suffix + '.json')).read_text()) for family, suffix in FAMILIES.items()}
        summary = compare_reports(baseline, after)
        seen = json.loads((reports / 'after-seen.json').read_text())
        same(before_seen, seen, COMMON + ['data_sha256', 'actual_exposures_sha256', 'selection', 'generation_budget'])
        identities = ['id', 'function', 'prompt_sha256', 'original_program_sha256', 'published_cases', 'target_tokens']
        if [[r[k] for k in identities] for r in before_seen['results']] != [[r[k] for k in identities] for r in seen['results']]:
            raise ValueError('Original source population changed')
        summary['seen_training_sources'] = {key: {'before': before_seen[key], 'after': seen[key]} for key in
            ['teacher_forced_nll', 'teacher_forced_token_accuracy', 'syntax_pass_rate', 'published_example_functional_rate']}
        summary['previously_correct_seen_sources_lost'] = [a['id'] for a, b in zip(before_seen['results'], seen['results'])
            if a['candidate']['functional_passed'] and not b['candidate']['functional_passed']]
        summary['single_mixed_data_continuation_not_isolated_augmentation_effect'] = True
        atomic_json(reports / 'comparison.json', summary)
        restore(archive); shutil.rmtree(location)
        d = json.loads((archive / 'manifest.json').read_text()); d['original_removed'] = True; atomic_json(archive / 'manifest.json', d)
        atomic_json(reports / 'temporary-cleanup.json', {'removed_owned_directory': str(location),
            'durable_archive': str(archive), 'exact_original_checkpoint_sha256': training['candidate_checkpoint_sha256'],
            'final_archive_manifest_sha256': file_hash(archive / 'manifest.json'), 'goal_achieved': False})
        status('binding/rehearsal training and matched assessments completed; broader capability unverified')
    except Exception as error:
        status('binding/rehearsal intervention stopped; requires inspection', error=str(error), temporary=str(location) if location else None)
        raise


if __name__ == '__main__': main()
