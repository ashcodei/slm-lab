"""Test balanced public instruction learning with persistent data and checkpoints."""
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch
from tokenizers import Tokenizer

from slm_lab.code_quality import evaluate as evaluate_code
from slm_lab.cpp_evaluation import evaluate as evaluate_cpp
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation import evaluate as evaluate_loss
from slm_lab.evaluation_provenance import capture
from slm_lab.instruction_generalization_trial import prepare
from slm_lab.knowledge_evaluation import evaluate as evaluate_knowledge
from slm_lab.model import ModelConfig, TinyMultimodalLM
from slm_lab.recurrent_ablation import frozen_digest, language_parameters, state_digest
from slm_lab.training import Encoder, supervised_gradients

ROOT = Path(__file__).resolve().parents[1]
STEPS = 1500
SEED = 88123


def main():
    torch.set_num_threads(1); torch.manual_seed(SEED)
    reports = ROOT / 'reports/instruction-generalization-probe-v2'
    if reports.exists(): raise ValueError('Inspect the existing instruction probe; never relaunch it blindly')
    reports.mkdir()
    def status(state, **fields):
        atomic_json(reports / 'status.json', {'status': state,
            'updated_at': datetime.now(timezone.utc).isoformat(), 'goal_achieved': False, **fields})
    source = ROOT / 'checkpoints/extended-curriculum-v1/language-pretraining-interrupted-1/step-00012000.pt'
    tokenizer = ROOT / 'data/tokenizer-corpus/tokenizer.json'
    files = list(dict.fromkeys([ROOT / name for name in capture()['files_sha256']]
        + [Path(__file__), ROOT / 'src/slm_lab/instruction_generalization_trial.py',
           ROOT / 'src/slm_lab/exercise_probe.py', ROOT / 'src/slm_lab/recurrent_ablation.py',
           ROOT / 'src/slm_lab/wikipedia_corpus.py']))
    runtime = capture(files)
    location = None
    try:
        status('binding exact retained own12000 foundation snapshot read-only')
        fd = os.open(source, os.O_RDONLY)
        try:
            before = os.fstat(fd)
            reference_path = Path(f'/proc/{os.getpid()}/fd/{fd}')
            source_sha = file_hash(reference_path)
            expected_source = json.loads((ROOT / 'reports/environment-restart-training-recovered-v1.json').read_text())
            if source_sha != expected_source['snapshot_sha256'] or source.is_symlink():
                raise ValueError('Retained own12000 foundation snapshot differs from recovery evidence')
            original = torch.load(reference_path, map_location='cpu', weights_only=True, mmap=True)
            stage = next(row for row in json.loads((ROOT / 'reports/extended-curriculum-v1/plan.json').read_text())['stages']
                         if row['name'] == 'language-pretraining')
            if (original['stage'] != 'pretrain' or original['steps'] != 12000
                    or original['planned_steps'] != stage['steps'] or original['data_sha256'] != stage['data_sha256']
                    or original['trainable'] != 'language' or original['world_size'] != 4
                    or original['truncated_sequences'] != 0):
                raise ValueError('Periodic source differs from the live own foundation stage')
            if ([row['step'] for row in original['metrics']] != list(range(1, 12001))
                    or any(row['global_target_tokens'] != 262080 for row in original['metrics'])):
                raise ValueError('Source12000 checkpoint is not the complete actual target budget')
            config = ModelConfig(**original['config']); token_sha = file_hash(tokenizer)
            if token_sha != original['tokenizer_sha256']: raise ValueError('Own checkpoint/tokenizer mismatch')
            with torch.device('meta'): count = TinyMultimodalLM(config).parameter_counts()['total']
            if count >= 1_000_000_000: raise ValueError('Model exceeds the user total parameter budget')
            bytes_needed = sum(value.numel() * value.element_size() for value in original['model'].values())
            if shutil.disk_usage(ROOT).free < bytes_needed + 26 * 1024**3:
                raise ValueError('Require candidate weights plus26GiB persistent reserve for live training')
            location = ROOT / 'checkpoints/instruction-generalization-probe-v2'
            location.mkdir(exist_ok=False)
            base = location / 'base/model.pt'; base.parent.mkdir()
            base_state = state_digest(original['model'])
            # Reuse the exact retained snapshot inode. Inference loads only its
            # model pages and discards optimizer state; no second8GiB copy.
            os.link(source, base)
            linked = base.stat()
            if (linked.st_dev, linked.st_ino) != (before.st_dev, before.st_ino):
                raise ValueError('Baseline link does not bind the inspected source inode')
            shutil.copy2(tokenizer, base.parent / 'tokenizer.json')
            after = os.fstat(fd)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError('Actual periodic source changed during weights export')
            del original; gc.collect()
        finally: os.close(fd)
        atomic_json(ROOT / 'reports/instruction-generalization-checkpoint-location-v2.json',
                    {'directory': str(location), 'reports': str(reports), 'goal_achieved': False})
        status('preparing complete original public instruction targets')
        pools, allocation, receipt = prepare(ROOT, location / 'data', tokenizer, config, STEPS, SEED)
        input_manifest = location / 'data/manifest.json'; receipt_sha = file_hash(input_manifest)
        plan = {'source_periodic_checkpoint_sha256': source_sha, 'source_periodic_steps': 12000,
            'base_checkpoint': str(base), 'base_checkpoint_sha256': file_hash(base), 'base_model_state_sha256': base_state,
            'own_total_parameters': count, 'tokenizer_sha256': token_sha, 'source_runtime': runtime,
            'base_storage': 'Hard link to exact retained periodic snapshot; read-only use, no optimizer-state duplication.',
            'persistent_output_directory': str(location),
            'input_manifest': str(input_manifest), 'input_manifest_sha256': receipt_sha,
            'source_pool_examples': receipt['source_pool_examples'], 'source_pool_exposures': receipt['source_pool_exposures'],
            'steps': STEPS, 'microbatch_size': 1, 'gradient_accumulation': 4,
            'planned_actual_response_targets': receipt['planned_actual_response_targets'],
            'learning_rate': 2e-5, 'weight_decay': .01, 'seed': SEED, 'device': 'cuda:3',
            'frozen': 'Original own vision and critic tensors', 'automatic_policy_promotion': False,
            'scope': receipt['scope'], 'goal_achieved': False,
            'limitations': ['Focused bounded public-data SFT, not the queued complete main curriculum.',
                'Reused development assessments; earlier source/concept exposure is not exhaustively excluded.',
                'Exact frozen vision tensors do not establish retained image behavior.',
                'Human demonstrations are original public data, not new ratings of this policy.']}
        atomic_json(reports / 'plan.json', plan)
        def guard():
            if capture(files) != runtime or file_hash(input_manifest) != receipt_sha:
                raise ValueError('Loaded instruction trial source/runtime/manifest changed')
            for path, digest in receipt['source_inputs_sha256'].items():
                if file_hash(path) != digest: raise ValueError('Admitted original source changed: ' + path)
            for name, digest in receipt['files_sha256'].items():
                if file_hash(input_manifest.parent / name) != digest: raise ValueError('Prepared exact source pool changed')
        device = 'cuda:3'; torch.cuda.set_per_process_memory_fraction(.5, 3)
        valid = ROOT / 'data/exercism-code-v1/valid.jsonl'
        questions = ROOT / 'data/knowledge-eval-v1/questions.jsonl'
        def assess(checkpoint, label):
            guard()
            for mode in ['completion', 'instruction']:
                status('assessing actual generated code for balanced instruction trial', policy=label, mode=mode)
                evaluate_code(checkpoint, reports / (label + '-code-' + mode + '.json'), device,
                              max_new_tokens=512, prompt_mode=mode)
                gc.collect(); torch.cuda.empty_cache()
            status('assessing native C++ instruction functions', policy=label)
            evaluate_cpp(checkpoint, reports / (label + '-cpp.json'), device, max_new_tokens=512)
            gc.collect(); torch.cuda.empty_cache()
            status('assessing full181 held module response loss', policy=label)
            evaluate_loss(checkpoint, valid, 'loss', reports / (label + '-module-loss.json'), device=device,
                          threads=1, max_examples=181)
            gc.collect(); torch.cuda.empty_cache()
            status('assessing all559 original knowledge questions', policy=label)
            evaluate_knowledge(checkpoint, questions, reports / (label + '-knowledge.json'), device, max_examples=559)
            gc.collect(); torch.cuda.empty_cache(); guard()
        assess(base, 'before')
        status('loading exact own foundation for balanced instruction learning')
        torch.cuda.set_per_process_memory_fraction(.5, 3)
        metadata = torch.load(base, weights_only=True, map_location='cpu', mmap=True)
        model = TinyMultimodalLM(config); model.load_state_dict(metadata['model']); del metadata; gc.collect()
        if state_digest(model.state_dict()) != base_state: raise ValueError('Exported own model tensor state differs')
        frozen = frozen_digest(model); parameters = language_parameters(model, False)
        model.to(device); model.gradient_checkpointing = True; model.train()
        encoder = Encoder(Tokenizer.from_file(str(tokenizer)), config, input_manifest.parent, 448)
        optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=.01, foreach=False)
        amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
        metrics = []; targets = 0; started = time.monotonic(); torch.cuda.reset_peak_memory_stats(3)
        for step, group in enumerate(allocation):
            optimizer.zero_grad(set_to_none=True)
            learning_rate = 2e-5 * min(1, (step + 1) / 150) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / STEPS)))
            for params in optimizer.param_groups: params['lr'] = learning_rate
            batches = [[pools[pick['pool']][pick['index']]] for pick in group]
            tick = time.monotonic()
            loss, count_targets = supervised_gradients(model, encoder, batches, 'response', device, amp)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True); optimizer.step()
            torch.cuda.synchronize(3); targets += count_targets
            metrics.append({'step': step + 1, 'loss': float(loss), 'grad_norm': float(norm),
                'actual_target_tokens': count_targets, 'learning_rate': learning_rate, 'seconds': time.monotonic() - tick})
            if (step + 1) % 25 == 0:
                guard(); status('training balanced original public coding instructions', completed_steps=step + 1,
                                planned_steps=STEPS, actual_response_targets=targets)
                print(json.dumps(metrics[-1]), flush=True)
        if targets != receipt['planned_actual_response_targets'] or encoder.truncated:
            raise ValueError('Actual response targets differ from complete public instruction plan')
        if frozen_digest(model) != frozen: raise ValueError('Balanced language SFT changed own vision/critic')
        optimizer.zero_grad(set_to_none=True); del optimizer; gc.collect(); torch.cuda.empty_cache()
        candidate = location / 'candidate/model.pt'; candidate.parent.mkdir()
        if shutil.disk_usage(location).free < bytes_needed + 26 * 1024**3:
            raise ValueError('Candidate output would exhaust the live training persistent reserve')
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}; state_sha = state_digest(state)
        torch.save({'config': dict(config.__dict__), 'model': state, 'stage': 'sft', 'steps': STEPS,
            'planned_steps': STEPS, 'tokenizer_sha256': token_sha, 'data_sha256': receipt_sha,
            'trainable': 'language', 'truncated_sequences': 0, 'model_state_sha256': state_sha,
            'source_base_checkpoint_sha256': plan['base_checkpoint_sha256'], 'source_runtime': runtime}, candidate)
        shutil.copy2(tokenizer, candidate.parent / 'tokenizer.json')
        atomic_json(reports / 'training.json', {'completed_steps': STEPS, 'steps': STEPS, 'planned_steps': STEPS,
            'status': 'training run completed; instruction generalization unverified', 'actual_response_targets': targets,
            'elapsed_seconds': time.monotonic() - started, 'metrics': metrics, 'frozen_vision_and_critic_sha256': frozen,
            'candidate_model_state_sha256': state_sha, 'candidate_checkpoint': str(candidate),
            'candidate_checkpoint_sha256': file_hash(candidate), 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(3),
            'goal_achieved': False})
        atomic_json(candidate.parent / 'training.json', {'steps': STEPS, 'planned_steps': STEPS,
            'status': 'training run completed; instruction generalization unverified',
            'full_training_report': str(reports / 'training.json'),
            'full_training_report_sha256': file_hash(reports / 'training.json'), 'goal_achieved': False})
        del state, model, parameters; gc.collect(); torch.cuda.empty_cache()
        assess(candidate, 'after'); guard()
        if file_hash(base) != plan['base_checkpoint_sha256']: raise ValueError('Frozen own baseline checkpoint changed')
        status('balanced public instruction learning and assessments completed; broad capability unverified',
               base_checkpoint=str(base), candidate_checkpoint=str(candidate), automatic_policy_promotion=False)
    except Exception as error:
        status('requires inspection; balanced instruction trial stopped', error=str(error), checkpoint_directory=str(location)); raise


if __name__ == '__main__': main()
