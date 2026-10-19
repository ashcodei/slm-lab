"""Same-source SFT versus anchored execution preferences on our own706M model."""
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

from slm_lab.anchored_preference_gradients_v1 import gradients, score
from slm_lab.checkpoint_xor import archive_checkpoint, restore
from slm_lab.code_quality import evaluate as code_evaluate
from slm_lab.corpus import file_hash
from slm_lab.cpp_evaluation import evaluate as cpp_evaluate
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation import evaluate as loss_evaluate
from slm_lab.evaluation_provenance import capture
from slm_lab.instruction_trial_comparison import compare_reports, same, COMMON
from slm_lab.knowledge_evaluation import evaluate as knowledge_evaluate
from slm_lab.model import ModelConfig, TinyMultimodalLM
from slm_lab.recurrent_ablation import frozen_digest, language_parameters, state_digest
from slm_lab.semantic_optimization_schedule_v1 import LANGUAGES, schedule, sequence_digest, validate_targets
from slm_lab.training import Encoder
from slm_lab.verified_code_diagnostics_v2 import evaluate as seen_evaluate

ROOT = Path(__file__).resolve().parents[1]
SEED = 88124
FAMILIES = {'code': 'code-completion', 'code_instruction': 'code-instruction',
            'cpp': 'cpp', 'knowledge': 'knowledge', 'module_loss': 'module-loss'}


def main():
    torch.set_num_threads(1)
    reports = ROOT / 'reports/semantic-code-optimization-v1'
    durable = ROOT / 'checkpoints/semantic-code-optimization-v1'
    if reports.exists() or durable.exists():
        raise ValueError('Inspect existing controlled trial; never relaunch automatically')
    reports.mkdir(); durable.mkdir()
    spec_path = ROOT / 'reports/semantic-code-optimization-next-experiment-v1.json'
    spec = json.loads(spec_path.read_text()); base = Path(spec['base_checkpoint'])
    pair_data = ROOT / spec['pairs']; source_manifest = ROOT / spec['source_pool_manifest']
    source = json.loads(source_manifest.read_text())
    pair_manifest = pair_data.parent / 'manifest.json'
    pairs = list(jsonl(pair_data)); device = 'cuda:3'
    valid = ROOT / 'data/exercism-code-v1/valid.jsonl'
    questions = ROOT / 'data/knowledge-eval-v1/questions.jsonl'
    bound = {path: file_hash(path) for path in [spec_path, base, pair_data, pair_manifest, source_manifest, valid, questions]}
    for name, digest in source['files_sha256'].items():
        path = source_manifest.parent / name
        if file_hash(path) != digest:
            raise ValueError('Original admitted instruction source changed')
        bound[path] = digest
    if (bound[base] != spec['base_checkpoint_sha256'] or bound[pair_data] != spec['pairs_sha256']
            or bound[pair_manifest] != spec['pairs_manifest_sha256'] or len(pairs) != 83
            or spec['updates_per_branch'] != 664 or spec['complete_pair_passes'] != 8):
        raise ValueError('Exact proposed experiment source/budget specification differs')
    files = list(dict.fromkeys([ROOT / name for name in capture()['files_sha256']] + [Path(__file__)] +
        [ROOT / 'src/slm_lab' / (name + '.py') for name in ['anchored_preference_gradients_v1',
         'semantic_optimization_schedule_v1', 'instruction_generalization_trial_v3',
         'instruction_trial_comparison', 'alignment_selection', 'checkpoint_xor', 'checkpoint_archive',
         'recurrent_ablation', 'verified_code_diagnostics_v2', 'vision_transfer', 'objectives']]))
    runtime = capture(files)

    def guard():
        if capture(files) != runtime:
            raise ValueError('Controlled trial implementation/runtime changed')
        for path, digest in bound.items():
            if file_hash(path) != digest:
                raise ValueError('Bound own policy or admitted source changed: ' + str(path))

    def status(message, **fields):
        atomic_json(reports / 'status.json', {'status': message,
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'automatic_policy_promotion': False, 'goal_achieved': False, **fields})

    def clean_cuda():
        gc.collect(); torch.cuda.empty_cache()

    location = None
    try:
        status('preparing exact same positive and rehearsal exposure')
        metadata = torch.load(base, map_location='cpu', weights_only=True, mmap=True)
        config = ModelConfig(**metadata['config']); tokenizer_path = base.parent / 'tokenizer.json'
        token_sha = file_hash(tokenizer_path); bound[tokenizer_path] = token_sha
        if (token_sha != metadata['tokenizer_sha256'] or metadata['steps'] != 1500
                or metadata['model_state_sha256'] != spec['base_model_state_sha256']):
            raise ValueError('Actual own base/tokenizer learning state differs')
        with torch.device('meta'):
            parameters = TinyMultimodalLM(config).parameter_counts()['total']
        if parameters != 706012672 or parameters >= 1_000_000_000:
            raise ValueError('Own model parameter budget differs')
        del metadata
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        encoder = Encoder(tokenizer, config, source_manifest.parent, 448)
        pools = {name: list(jsonl(source_manifest.parent / (name + '.jsonl')))
                 for name in [*LANGUAGES, 'human_instruction']}
        allocation = schedule(pairs, pools, passes=8, seed=SEED)
        preparation = validate_targets(pairs, pools, allocation, encoder)
        if len(allocation) != 664:
            raise ValueError('Scheduled source exposure differs from exact branch budget')
        exposures = reports / 'exposures.jsonl'
        exposures.write_text(''.join(json.dumps(row) + '\n' for row in preparation.pop('exposures')))
        bound[exposures] = file_hash(exposures)
        atomic_json(reports / 'preparation.json', preparation)
        bound[reports / 'preparation.json'] = file_hash(reports / 'preparation.json')
        plan = {**spec, 'status': 'controlled trial prepared', 'seed': SEED,
            'source_runtime': runtime, 'own_total_parameters': parameters,
            'tokenizer_sha256': token_sha, 'exposures_sha256': bound[exposures],
            'preparation_sha256': bound[reports / 'preparation.json'],
            'planned_targets_per_branch': preparation['planned_target_exposures'],
            'all_complete_masks_verified': True, 'training_started': False,
            'limitations': ['Same positive/rehearsal exposure, additional negative compute in preference branch.',
                'Reused development assessments; no independent final or frontier comparison.',
                'Frozen vision/critic tensors do not establish preserved image behavior.',
                'Own synthetic execution preferences are distinct from genuine human RLHF.']}
        atomic_json(reports / 'plan.json', plan)
        bound[reports / 'plan.json'] = file_hash(reports / 'plan.json')

        baseline = {}
        for family, suffix in FAMILIES.items():
            path = ROOT / 'reports/instruction-generalization-probe-v3' / ('after-' + suffix + '.json')
            report = json.loads(path.read_text())
            if report['checkpoint_sha256'] != bound[base] or report['evaluation_provenance'] != capture():
                raise ValueError('Exact full own-base assessment/runtime differs')
            baseline[family] = report; shutil.copy2(path, reports / ('before-' + suffix + '.json'))
            bound[path] = file_hash(path)
        compare_reports(baseline, baseline)
        if baseline['module_loss']['data_sha256'] != bound[valid] or baseline['knowledge']['questions_sha256'] != bound[questions]:
            raise ValueError('Original full held assessment inputs differ')
        before_seen_path = ROOT / 'reports/exposed-code-learning-after-instruction-v1/after.json'
        before_seen = json.loads(before_seen_path.read_text())
        if (before_seen['checkpoint_sha256'] != bound[base] or before_seen['examples'] != 128
                or capture([ROOT / path for path in before_seen['evaluation_provenance']['files_sha256']]) != before_seen['evaluation_provenance']):
            raise ValueError('Own base source diagnostic differs')
        shutil.copy2(before_seen_path, reports / 'before-seen.json'); bound[before_seen_path] = file_hash(before_seen_path)
        guard()
        torch.cuda.set_per_process_memory_fraction(.5, 3)
        if shutil.disk_usage('/tmp').free < 14 * 1024**3:
            raise ValueError('Require owned temporary checkpoint/archive capacity')
        location = Path(tempfile.mkdtemp(prefix='slm-semantic-optimization-v1-', dir='/tmp'))
        atomic_json(reports / 'temporary-location.json', {'directory': str(location),
            'scope': 'Temporary full files; completed policies require verified durable archives.', 'goal_achieved': False})
        amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)

        def load_model():
            torch.manual_seed(SEED)
            state = torch.load(base, map_location='cpu', weights_only=True, mmap=True)
            model = TinyMultimodalLM(config); model.load_state_dict(state['model']); del state
            if state_digest(model.state_dict()) != spec['base_model_state_sha256']:
                raise ValueError('Actual branch initialization differs from exact own base')
            language_parameters(model, False); model.to(device)
            model.gradient_checkpointing = True
            return model

        status('caching exact frozen own reference response scores')
        model = load_model(); model.eval(); references = {}
        with torch.no_grad():
            for row in pairs:
                good, bad = [encoder.batch([row], field, device) for field in ['chosen', 'rejected']]
                with amp():
                    a, b = score(model, good), score(model, bad)
                references[row['id']] = {'chosen': float(a), 'rejected': float(b),
                    'chosen_sequence_sha256': sequence_digest(encoder, row, 'chosen'),
                    'rejected_sequence_sha256': sequence_digest(encoder, row, 'rejected')}
        frozen = frozen_digest(model); del model; clean_cuda()
        reference_path = reports / 'reference-scores.json'
        atomic_json(reference_path, {'own_reference_checkpoint_sha256': bound[base],
            'tokenizer_sha256': token_sha, 'scores': references, 'goal_achieved': False})
        bound[reference_path] = file_hash(reference_path); guard()
        completed = {}
        for branch in ['control', 'anchored']:
            status('training controlled own-code optimization', branch=branch, completed_steps=0, planned_steps=664)
            model = load_model(); model.train()
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=2e-5, weight_decay=.01, foreach=False)
            metrics = []; totals = {name: 0 for name in ['chosen', 'rejected', 'module', 'human']}
            started = time.monotonic(); torch.cuda.reset_peak_memory_stats(3)
            for step, pick in enumerate(allocation):
                optimizer.zero_grad(set_to_none=True)
                lr = 2e-5 * min(1, (step + 1) / 64) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / 664)))
                for group in optimizer.param_groups: group['lr'] = lr
                pair = pairs[pick['pair_index']]; good = encoder.batch([pair], 'chosen', device)
                tick = time.monotonic()
                if branch == 'anchored':
                    bad = encoder.batch([pair], 'rejected', device); reference = references[pair['id']]
                    actual = gradients(model, good, bad,
                        torch.tensor([reference['chosen']], device=device, dtype=torch.float32),
                        torch.tensor([reference['rejected']], device=device, dtype=torch.float32), amp)
                    base_loss = actual['loss']; totals['rejected'] += actual['rejected_target_tokens']
                else:
                    with amp(): loss = model(**good)['loss']
                    if not torch.isfinite(loss): raise ValueError('Nonfinite control positive loss')
                    loss.backward(); base_loss = float(loss.detach())
                    actual = {'chosen_nll': base_loss, 'chosen_target_tokens': int((good['labels'][:, 1:] != -100).sum())}
                totals['chosen'] += actual['chosen_target_tokens']
                rehearsal = {}
                for name, row in [('module', pools[pick['module_pool']][pick['module_index']]),
                                  ('human', pools['human_instruction'][pick['human_index']])]:
                    batch = encoder.batch([row], 'response', device)
                    with amp(): loss = model(**batch)['loss']
                    if not torch.isfinite(loss): raise ValueError('Nonfinite rehearsal loss')
                    (.25 * loss).backward(); rehearsal[name + '_nll'] = float(loss.detach())
                    totals[name] += int((batch['labels'][:, 1:] != -100).sum())
                    del loss
                norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1, error_if_nonfinite=True))
                optimizer.step(); torch.cuda.synchronize(3)
                metrics.append({'step': step + 1, 'pair_id': pair['id'], 'learning_rate': lr,
                    **actual, **rehearsal, 'pair_objective': base_loss,
                    'loss': base_loss + .25 * sum(rehearsal.values()),
                    'gradient_norm_before_clipping': norm, 'step_seconds': time.monotonic() - tick})
                if step == 0 or (step + 1) % 10 == 0 or step == 663:
                    print(json.dumps({'branch': branch, **metrics[-1], 'actual_target_exposures': totals}), flush=True)
                    status('training controlled own-code optimization', branch=branch,
                           completed_steps=step + 1, planned_steps=664, actual_target_exposures=totals)
                if (step + 1) % 100 == 0: guard()
            expected = preparation['planned_target_exposures']
            if encoder.truncated or any(totals[name] != expected[name] for name in ['chosen', 'module', 'human']):
                raise ValueError('Actual matched positive/rehearsal targets differ')
            if totals['rejected'] != (expected['rejected'] if branch == 'anchored' else 0) or frozen_digest(model) != frozen:
                raise ValueError('Actual negative exposure or frozen vision/critic differs')
            path = location / branch / 'model.pt'; path.parent.mkdir()
            state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
            state_sha = state_digest(state)
            torch.save({'config': dict(config.__dict__), 'model': state, 'stage': 'sft' if branch == 'control' else 'anchored_execution_preference',
                'steps': 664, 'planned_steps': 664, 'tokenizer_sha256': token_sha, 'data_sha256': bound[exposures],
                'trainable': 'language', 'truncated_sequences': 0, 'model_state_sha256': state_sha,
                'source_base_checkpoint_sha256': bound[base], 'feedback_source': 'synthetic' if branch == 'anchored' else None}, path)
            shutil.copy2(tokenizer_path, path.parent / 'tokenizer.json')
            branch_report = reports / branch; branch_report.mkdir()
            training = {'status': 'controlled branch training completed; capability unverified',
                'steps': 664, 'planned_steps': 664, 'actual_target_exposures': dict(totals), 'metrics': metrics,
                'base_checkpoint_sha256': bound[base], 'candidate_checkpoint_sha256': file_hash(path),
                'candidate_model_state_sha256': state_sha, 'frozen_vision_and_critic_sha256': frozen,
                'elapsed_seconds': time.monotonic() - started, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(3),
                'goal_achieved': False, 'automatic_policy_promotion': False}
            atomic_json(branch_report / 'training.json', training)
            atomic_json(path.parent / 'training.json', {'status': training['status'], 'steps': 664, 'planned_steps': 664})
            del state, model, trainable, optimizer, good, batch; clean_cuda()
            status('preserving completed branch in verified durable archive', branch=branch)
            staged = location / branch / 'model.pt.xor'
            result = archive_checkpoint(path, base, staged, remove_original=False)
            if shutil.disk_usage(ROOT).free < result['payload_bytes'] + 26 * 1024**3:
                raise ValueError('Completed temporary policy retained; durable archive would violate26GiB reserve')
            destination = durable / branch / 'model.pt.xor'; destination.parent.mkdir()
            copying = destination.with_name(destination.name + '.staging')
            shutil.copytree(staged, copying)
            for item in copying.iterdir():
                with item.open('rb') as stream: os.fsync(stream.fileno())
            verification = restore(copying)
            copying.rename(destination)
            atomic_json(branch_report / 'durable-archive.json', {'archive': str(destination),
                'manifest_sha256': file_hash(destination / 'manifest.json'), 'verification': verification,
                'temporary_full_checkpoint': str(path), 'mandatory_full_anchor': str(base), 'goal_achieved': False})
            completed[branch] = {'path': path, 'archive': destination, 'checkpoint_sha256': training['candidate_checkpoint_sha256']}
            guard()

        for branch, candidate in completed.items():
            path = candidate['path']; folder = reports / branch
            if file_hash(path) != candidate['checkpoint_sha256']: raise ValueError('Completed temporary policy changed')
            for mode in ['completion', 'instruction']:
                status('assessing matched controlled branch', branch=branch, family='code', mode=mode)
                code_evaluate(path, folder / ('after-code-' + mode + '.json'), device, max_new_tokens=512, prompt_mode=mode)
                clean_cuda(); guard()
            status('assessing matched controlled branch', branch=branch, family='native C++')
            cpp_evaluate(path, folder / 'after-cpp.json', device, max_new_tokens=512); clean_cuda(); guard()
            status('assessing matched controlled branch', branch=branch, family='full181 module response loss')
            loss_evaluate(path, valid, 'loss', folder / 'after-module-loss.json', device=device, threads=1, max_examples=181)
            clean_cuda(); guard()
            status('assessing matched controlled branch', branch=branch, family='full559 knowledge questions')
            knowledge_evaluate(path, questions, folder / 'after-knowledge.json', device, max_examples=559)
            clean_cuda(); guard()
            status('assessing matched controlled branch', branch=branch, family='original128 exposed-source programs')
            seen_evaluate(path, source_manifest.parent / 'verified_python.jsonl', folder / 'after-seen.json',
                          device, count=128, max_new_tokens=512, exposures=source_manifest.parent / 'exposures.jsonl')
            clean_cuda(); guard()
            after = {family: json.loads((folder / ('after-' + suffix + '.json')).read_text()) for family, suffix in FAMILIES.items()}
            summary = compare_reports(baseline, after)
            seen = json.loads((folder / 'after-seen.json').read_text())
            same(before_seen, seen, COMMON + ['data_sha256', 'actual_exposures_sha256', 'selection', 'generation_budget'])
            identities = ['id', 'prompt_sha256', 'original_program_sha256', 'published_cases', 'target_tokens']
            if [[row[key] for key in identities] for row in before_seen['results']] != [[row[key] for key in identities] for row in seen['results']]:
                raise ValueError('Original exposed-source population differs')
            summary['seen_training_sources'] = {key: {'before': before_seen[key], 'after': seen[key]} for key in
                ['teacher_forced_nll', 'teacher_forced_token_accuracy', 'syntax_pass_rate', 'published_example_functional_rate']}
            summary['durable_candidate_archive'] = str(candidate['archive'])
            atomic_json(folder / 'comparison.json', summary)
        control = {family: json.loads((reports / 'control' / ('after-' + suffix + '.json')).read_text()) for family, suffix in FAMILIES.items()}
        anchored = {family: json.loads((reports / 'anchored' / ('after-' + suffix + '.json')).read_text()) for family, suffix in FAMILIES.items()}
        result = compare_reports(control, anchored)
        result.update(scope='Anchored versus same-positive/rehearsal supervised control; extra negative compute, not equal-compute proof.',
            plan_sha256=bound[reports / 'plan.json'], completed_at=datetime.now(timezone.utc).isoformat(),
            image_behavior_assessed=False, human_feedback_preference_assessed=False)
        atomic_json(reports / 'controlled-comparison.json', result)
        guard()
        for branch, candidate in completed.items():
            restore(candidate['archive'])
            shutil.rmtree(location / branch)
            manifest = json.loads((candidate['archive'] / 'manifest.json').read_text())
            manifest['original_removed'] = True
            atomic_json(candidate['archive'] / 'manifest.json', manifest)
            atomic_json(reports / branch / 'temporary-cleanup.json', {'removed_owned_directory': str(location / branch),
                'durable_archive': str(candidate['archive']), 'exact_original_checkpoint_sha256': candidate['checkpoint_sha256'],
                'final_archive_manifest_sha256': file_hash(candidate['archive'] / 'manifest.json'), 'goal_achieved': False})
        location.rmdir()
        status('controlled semantic optimization and matched assessments completed; broader capability unverified')
    except Exception as error:
        status('controlled semantic optimization stopped; requires inspection', error=str(error),
               retained_temporary_directory=str(location) if location is not None else None)
        raise


if __name__ == '__main__':
    main()
