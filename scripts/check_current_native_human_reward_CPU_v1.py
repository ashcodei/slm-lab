"""Real984M/current-source feature and disposable reward-gradient diagnostic.

Use one complete shortest training pair and one complete shortest held pair.
Only the training pair supplies one disposable head update. This is not a
full human epoch, PPO training, capacity admission or a model-quality result.
The standalone source checkpoint and all original GPU workers stay unchanged.
"""
from datetime import datetime, timezone
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/tmp/slm-completed-joint-initializer-CPU-v3')
OUT = Path('/tmp/slm-current-native-human-reward-CPU-v1')


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.local_source_closure_v2 import python_sources
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('A fresh CUDA-hidden nice19 single-thread disposable CPU diagnostic is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    source = read(SOURCE / 'complete.json'); source_plan = read(SOURCE / 'plan.json')
    tests = ROOT / 'reports/native-human-reward-v1-tests-final.log'
    if (not terminal(source_plan['owner']) or (SOURCE / 'failure.json').exists() or source.get('complete') is not True
            or source.get('actual_counts', {}).get('total_unique') != 984399690
            or source.get('all_native_values_and_actual_text_photo_word_cold_outputs_exact') is not True
            or '7 passed' not in tests.read_text()):
        raise ValueError('The actual completed full native and the unchanged reward source/gradient/recovery checks must precede diagnostics')
    files = {str(p): sha(p) for p in [SOURCE / 'plan.json', SOURCE / 'complete.json', tests,
        *python_sources(ROOT, [Path(__file__)])]}
    files[source['checkpoint']] = source['checkpoint_sha256']
    files[str(Path(source['checkpoint']).parent / 'tokenizer.json')] = source['tokenizer_sha256']
    OUT.mkdir(); owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    def write(name, value):
        from slm_lab.reserved_atomic_metadata_v1 import write_json
        return write_json(OUT / name, value, root=OUT, minimum_free_bytes=40 * 1024**3,
            maximum_metadata_bytes=16 * 1024**2, guard=lambda: None)
    write('plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        exact_completed_native_owner=source_plan['owner'], actual_native_values=984399690,
        declared_scope='one whole train pair/one whole held pair; one disposable train-only reward-head update',
        shortest_complete_source_selection_is_CPU_diagnostic_only=True, whole_source_training_and_PPO_not_admitted=True,
        no_production_initializer_or_benchmark_changed=True, GPU_contexts=0, goal_achieved=False))
    try:
        import torch
        from torch.nn import functional as F
        from slm_lab.codenet_checkpoint_v1 import FileBindings, capture_rng
        from slm_lab.native_human_reward_v1 import native_features, signature
        from slm_lab.own_dense_standalone_v1 import load
        from slm_lab.parallel_function_repair_stage_v1 import value_digest
        from slm_lab.whole_human_data_v2 import WholeHumanData
        from slm_lab.whole_human_reward_v1 import source_fingerprint
        from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
        guard = FileBindings(files); write('progress.json', dict(phase='loading actual completed984Mnative on CPU'))
        base, metadata = load(source['checkpoint'], expected_checkpoint_sha256=source['checkpoint_sha256'], expected_sources=source['sources'])
        base.eval()
        if base.counts()['total_unique'] != 984399690 or torch.cuda.is_initialized(): raise ValueError('Only the actual full CPU native is eligible')
        data_root = ROOT / 'data/whole-human-alignment-v2'
        train = WholeHumanData(data_root, 'preferences', 'train', 'optimization')
        held = WholeHumanData(data_root, 'preferences', 'valid', 'evaluation')
        if len(train) != 85786 or len(held) != 4642: raise ValueError('Every original train/held population must remain intact')
        before_rng = value_digest(capture_rng('cpu')); features = {}; sources = {}; train_ids = None
        for label, data in [('train', train), ('held', held)]:
            position = min(range(len(data)), key=lambda p: sum(data.rows[p]['native'][f]['length'] for f in ['chosen', 'rejected']))
            write('progress.json', dict(phase='encoding whole actual original human pair', role=label, original_position=position))
            values = []
            for field in ['chosen', 'rejected']:
                ids, labels = data.sequence(position, field); batch = torch.tensor([ids], dtype=torch.long)
                values.append(native_features(base, batch)[0])
                if label == 'train' and field == 'chosen': train_ids = batch
            features[label] = torch.stack(values)
            sources[label] = dict(original_position=position, role=data.role, purpose=data.purpose,
                source_fingerprint=source_fingerprint(data, position), id=data.rows[position]['id'],
                complete_sequence_lengths={f: data.rows[position]['native'][f]['length'] for f in ['chosen', 'rejected']})
        head = base.heads['reward']; parameters = list(head.parameters()); ids = {id(p) for p in parameters}
        fixed = [(n, p) for n, p in base.named_parameters() if id(p) not in ids]
        fixed_signature, fixed_digest = signature(p for _, p in fixed), tensor_digest(fixed)
        original_head = copy.deepcopy(head.state_dict()); original_head_sha = tensor_digest(head.named_parameters())
        oracle = copy.deepcopy(head)
        setting = dict(lr=1e-3, betas=(.9, .95), weight_decay=.01, foreach=False)
        optimizer = torch.optim.AdamW(parameters, **setting); oracle_optimizer = torch.optim.AdamW(oracle.parameters(), **setting)
        with torch.no_grad(): original_logits = base(train_ids)['logits'].clone()
        pair = head(features['train']).squeeze(-1); loss = F.softplus(pair[1]-pair[0])
        loss.backward(inputs=parameters)
        predicted = oracle(features['train']).squeeze(-1)
        expected_loss = torch.logaddexp(torch.zeros(()), predicted[1]-predicted[0]); expected_loss.backward()
        if (not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters)
                or any(p.grad is not None for _, p in fixed) or features['held'].requires_grad):
            raise ValueError('Only the actual current native reward head may receive the train-only finite gradients')
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
        torch.nn.utils.clip_grad_norm_(oracle.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True); oracle_optimizer.step()
        maximum_head_error = max(float((p-oracle.get_parameter(n)).abs().max()) for n, p in head.named_parameters())
        with torch.no_grad():
            logits = base(train_ids)['logits']; held_scores = head(features['held']).squeeze(-1)
        if (maximum_head_error > 2e-7 or not torch.equal(original_logits, logits)
                or signature(p for _, p in fixed) != fixed_signature or tensor_digest(fixed) != fixed_digest
                or tensor_digest(head.named_parameters()) == original_head_sha or len(optimizer.state) != 3
                or any(float(s['step']) != 1 for s in optimizer.state.values())
                or value_digest(capture_rng('cpu')) != before_rng or torch.cuda.is_initialized()):
            raise ValueError('Actual full-size train-only head learning, scalar oracle, fixed native values/logits and CPU RNG must reconcile')
        artifact = OUT / 'disposable-real-source-head-diagnostic.pt'
        torch.save(dict(format='own-current-native-real-source-head-CPU-diagnostic-only-v1', original_sources=sources,
            detached_actual_native_features={k: v.detach().cpu() for k, v in features.items()},
            before_head=original_head, after_head=head.state_dict(), optimizer=optimizer.state_dict(),
            production_head_or_PPO_admission=False), artifact)
        guard.check(full=True)
        write('complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
            actual_native_values=984399690, actual_reward_head_values=sum(p.numel() for p in parameters), actual_Adam_states=3,
            actual_disposable_reward_head_updates=1, actual_source_pairs=sources, actual_FP32_feature_shapes={k: list(v.shape) for k, v in features.items()},
            maximum_independent_scalar_oracle_head_error=maximum_head_error, actual_gradient_norm_before_clip=norm,
            native_language_vision_critic_word_auxiliary_values_and_parameter_metadata_unchanged=True,
            actual_complete_train_text_logits_unchanged=True, original_CPU_RNG_unchanged=True,
            actual_held_scores_diagnostic_only=held_scores.tolist(), held_gradients=False,
            diagnostic_artifact=str(artifact), diagnostic_artifact_sha256=sha(artifact), diagnostic_excluded_from_production_initialization=True,
            full85786pair_reward_epoch_4642held_ranking_and_actual_PPO_training_still_required=True,
            no_source_checkpoint_or_existing_worker_changed=True, GPU_contexts=0, model_quality_gain=False, goal_achieved=False))
        print(json.dumps(dict(complete=True, actual_native_values=984399690, actual_reward_head_values=sum(p.numel() for p in parameters))), flush=True)
    except BaseException as exc:
        write('failure.json', dict(owner=owner, error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False,
            actual_source_and_all_partial_outputs_preserved=True, goal_achieved=False)); raise


if __name__ == '__main__': main()
