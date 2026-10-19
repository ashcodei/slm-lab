"""Actual full-size frozen-general human reward features and head recovery.

Two complete original training pairs test the full model computation. These
disposable updates are not a full ranking pass or a reward-quality result.
"""
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/whole-human-reward-full-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only full-size probe requires explicitly hidden CUDA devices')
    if REPORTS.exists():
        raise ValueError('Preserve every previous attempt; no automatic probe restart')
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings, atomic_checkpoint, capture_rng
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.own_alignment_checkpoint_v1 import make_state, restore_state
    from slm_lab.point_grounding_evaluation_v1 import state_digest
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.whole_human_reward_v1 import general_features, scores, pairwise_loss, source_fingerprint, _RewardRestoreView
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    from slm_lab.whole_human_sft_inference_v1 import fresh
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    source_plan = ROOT/'reports/whole-human-SFT-v1/plan.json'
    old = json.loads(source_plan.read_text())
    files = dict(old['source_files_sha256'])
    for name in ('scripts/probe_whole_human_reward_full_CPU_v1.py',
                 'src/slm_lab/whole_human_reward_v1.py', 'tests/test_whole_human_reward_v1.py',
                 'reports/whole-human-reward-v1-tests.log'):
        files[str(ROOT/name)] = file_hash(ROOT/name)
    files[str(source_plan)] = file_hash(source_plan)
    guard = FileBindings(files); guard.check(full=True)
    if '3 passed' not in (ROOT/'reports/whole-human-reward-v1-tests.log').read_text():
        raise ValueError('Actual frozen encoder, whole-source feature, reward-only recovery tests required')
    data = WholeHumanData(ROOT/'data/whole-human-alignment-v2', 'preferences', 'train', 'optimization')
    positions = [0, max(range(len(data)), key=lambda p: sum(data.rows[p]['native'][f]['length'] for f in ('chosen', 'rejected')))]
    REPORTS.mkdir()
    atomic_json(REPORTS/'plan.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=files, source_positions=positions, full_original_ranking_pass=False,
        source_plan_sha256=file_hash(source_plan), actual_counts=old['actual_counts'], goal_achieved=False))
    started = time.monotonic(); model, _ = fresh(old, 'cpu')
    selected = model.extension_parameters('reward'); selected_ids = {id(p) for p in selected}
    fixed = [p for p in model.parameters() if id(p) not in selected_ids]
    signatures = [(id(p), p._version, p.data_ptr()) for p in fixed]
    original_digest = state_digest(model.original)
    def fixed_parts():
        return tensor_digest((n, p) for n, p in model.parts.named_parameters() if not n.startswith('reward.'))
    unused_digest = fixed_parts()
    ids = torch.tensor([[1, 4, 2]])
    with torch.no_grad():
        native = {route:model(ids, route=route)['logits'].clone() for route in ('general', 'code', 'grounding')}
        cache = model.original.language.code.prefill(ids)['cache']
        cached = model.original.language.code.decode(torch.tensor([[5]]), cache)['logits'].clone()
    features = []; rows = []
    for p in positions:
        values = []
        for field in ('chosen', 'rejected'):
            source, _ = data.sequence(p, field)
            values.append(general_features(model, torch.tensor([source]))[0])
            print('complete_original_human_response_encoded', p, field, len(source), flush=True)
        features.append(torch.stack(values))
        rows.append(dict(position=p, id=data.rows[p]['id'], source_fingerprint=source_fingerprint(data, p),
            native=data.rows[p]['native']))
    features = torch.stack(features)
    optimizer = torch.optim.AdamW(selected, lr=1e-4, betas=(.9, .95), weight_decay=.01, foreach=False)
    def update():
        optimizer.zero_grad(set_to_none=True)
        values = scores(model, features); loss = pairwise_loss(values); loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in selected) or any(p.grad is not None for p in fixed):
            raise ValueError('Every and only actual full reward-head tensor requires finite gradients')
        norm = float(torch.nn.utils.clip_grad_norm_(selected, 1., error_if_nonfinite=True))
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        return dict(pair_mean_nll=float(loss.detach()), gradient_norm_before_clip=norm)
    sources = dict(probe_plan_sha256=file_hash(REPORTS/'plan.json'), source_plan_sha256=file_hash(source_plan))
    first = update()
    state = make_state(model, optimizer, groups=('reward',), sources=sources, completed_steps=1,
        planned_steps=2, sampler_state=dict(capacity_only_source_positions=positions, completed_updates=1), metrics=[first], device='cpu')
    atomic_checkpoint(REPORTS/'resume.pt', state, 64*1024**2, 26*1024**3)
    expected = update(); expected_parts = model.extension_state(); expected_adam = copy.deepcopy(optimizer.state_dict())
    rng = capture_rng('cpu')
    restored = torch.load(REPORTS/'resume.pt', map_location='cpu', weights_only=True, mmap=True)
    restore_state(_RewardRestoreView(model), optimizer, restored, groups=('reward',), expected_sources=sources,
        planned_steps=2, expected_sampler_state=dict(capacity_only_source_positions=positions, completed_updates=1), device='cpu')
    if update() != expected or any(not torch.equal(t, model.parts.state_dict()[n]) for n, t in expected_parts.items()):
        raise ValueError('Actual next reward update and every extension tensor must recover exactly')
    def equal(a, b):
        if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
        if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(equal(v, b[k]) for k, v in a.items())
        if isinstance(a, (list, tuple)): return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b
    if not equal(expected_adam, optimizer.state_dict()) or not equal(rng, capture_rng('cpu')):
        raise ValueError('Every reward Adam and Python/NumPy/CPU RNG byte must recover exactly')
    if (signatures != [(id(p), p._version, p.data_ptr()) for p in fixed]
            or original_digest != state_digest(model.original) or unused_digest != fixed_parts()):
        raise ValueError('All original and unused new tensors, versions and pointers must remain fixed')
    with torch.no_grad():
        if any(not torch.equal(native[r], model(ids, route=r)['logits']) for r in native):
            raise ValueError('Every original route must remain exact')
        if not torch.equal(cached, model.original.language.code.decode(torch.tensor([[5]]), cache)['logits']):
            raise ValueError('Pre-existing original code cache must remain valid and exact')
    guard.check(full=True)
    atomic_json(REPORTS/'complete.json', dict(actual_counts=model.counts(),
        actual_reward_parameters=sum(p.numel() for p in selected), actual_reward_Adam_states=len(optimizer.state),
        actual_disposable_logical_updates=2, physical_update_calls=3, original_sources=rows,
        complete_context_tokens=sum(r['native'][f]['length'] for r in rows for f in ('chosen', 'rejected')),
        frozen_original_general_encoder_sha256=state_digest(model.original.language.base),
        fixed_original_digest=original_digest, all_unused_new_extensions_digest=unused_digest,
        actual_next_update_all_reward_tensors_Adam_cursor_and_RNG_exact=True,
        all_fixed_bytes_versions_pointers_original_native_routes_and_code_cache_exact=True,
        initial_and_next_update_metrics=[first, expected], resume_sha256=file_hash(REPORTS/'resume.pt'),
        seconds=time.monotonic()-started, full_original_ranking_pass=False, reward_quality_measured=False,
        all_capacity_updates_to_be_discarded=True, candidate_promoted=False, goal_achieved=False))


if __name__ == '__main__':
    main()
