"""Count the actual full expert graph and check one complete train-only update.

One disposable CPU update proves graph isolation, not training effectiveness.
It neither changes the live full-source run nor loads an external model.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/own-task-experts-full-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or REPORTS.exists():
        raise ValueError('Fresh CPU-only proof required')
    import torch
    from slm_lab.codenet_token_pack_v1 import CompleteCodePack
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.fast_joint_inference_v2 import fresh
    from slm_lab.own_task_experts_v1 import OwnTaskExperts, EXPERTS
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic(); REPORTS.mkdir()
    prior = ROOT / 'reports/phase-joint-v3/plan.json'
    plan = json.loads(prior.read_text())
    files = [Path(__file__), prior, Path(plan['initial_dense_settings']['initializer']),
        ROOT / 'src/slm_lab/own_task_experts_v1.py', ROOT / 'tests/test_own_task_experts_v1.py',
        ROOT / 'reports/own-task-experts-v1-tests.log']
    pack = CompleteCodePack(ROOT / 'data/verified-code-views-v1', 'train', 'training')
    files += [pack.folder / 'manifest.json', pack.folder / 'admission.json']
    files += [pack.folder / name for name in pack.manifest['files_sha256']]
    bindings = {str(path): file_hash(path) for path in files}
    position = min(range(len(pack)), key=lambda p: int(pack.index[pack.positions[p]]['length']))
    ids_, labels_ = pack.sequence(position)
    atomic_json(REPORTS / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=bindings, CPU_threads=1, CUDA_visible_devices='',
        complete_train_role_position=position, context_positions=len(ids_),
        targets=sum(x != -100 for x in labels_), disposable_optimizer_updates=1,
        deployed_or_production_training_updates=0, capability_gain_not_tested=True, goal_achieved=False))
    base, _ = fresh(plan, 'cpu')
    base_before = tensor_digest(base.named_parameters())
    base_metadata = [(id(p), p._version, p.data_ptr(), p.requires_grad) for p in base.parameters()]
    rng = torch.get_rng_state().clone()
    model = OwnTaskExperts(base).eval()
    if not torch.equal(rng, torch.get_rng_state()) or model.counts()['actual_unique'] != 999924554:
        raise ValueError('Actual strict sub-1B graph and private constructor RNG preservation required')
    ids = torch.tensor([ids_]); labels = torch.tensor([labels_])
    before = {name: tensor_digest(model.experts[name].parts.named_parameters()) for name in EXPERTS}
    with torch.no_grad():
        own_logits = base(ids)['logits']
        for name in EXPERTS:
            if not torch.equal(own_logits, model(ids, expert=name)['logits']):
                raise ValueError('Every initial expert must preserve own full-backbone text computation')
            if sum(p.numel() for p in model.native_view(name).parameters()) != 999924554:
                raise ValueError('Native selected view must register all expert/head parameters')
        assistant_before = model(ids, expert='assistant')['logits'].clone()
        reasoning_before = model(ids, expert='reasoning')['logits'].clone()
        base_loss = float(model(ids, labels=labels, expert='code')['loss'])
    selected = model.named_selected('code'); parameters = [p for _, p in selected]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, betas=(.9, .95), weight_decay=.01, foreach=False)
    model.train(); model.experts['code'].policy.gradient_checkpointing = True
    loss = model(ids, labels=labels, expert='code')['loss']
    loss.backward()
    if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters)
            or any(p.grad is not None for p in base.parameters()) or model.critic.weight.grad is not None
            or any(p.grad is not None for name in ['assistant', 'reasoning'] for _, p in model.named_selected(name))):
        raise ValueError('Only selected expert must receive finite gradients')
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
    optimizer.step(); optimizer.zero_grad(set_to_none=True); model.eval()
    if len(optimizer.state) != len(parameters) or any(float(s['step']) != 1 for s in optimizer.state.values()):
        raise ValueError('Every selected actual Adam clock must be one')
    after = {name: tensor_digest(model.experts[name].parts.named_parameters()) for name in EXPERTS}
    if (before['code'] == after['code'] or any(before[name] != after[name] for name in ['assistant', 'reasoning'])
            or base_before != tensor_digest(base.named_parameters())
            or base_metadata != [(id(p), p._version, p.data_ptr(), p.requires_grad) for p in base.parameters()]):
        raise ValueError('Base and unselected experts must remain exactly unchanged')
    with torch.no_grad():
        if (not torch.equal(assistant_before, model(ids, expert='assistant')['logits'])
                or not torch.equal(reasoning_before, model(ids, expert='reasoning')['logits'])):
            raise ValueError('Unselected actual full-size text computations changed')
        after_loss = float(model(ids, labels=labels, expert='code')['loss'])
    sources = dict(own_base_plan_sha256=file_hash(prior), train_source_admission_sha256=file_hash(pack.folder / 'admission.json'))
    payload = model.export(sources)
    cold = OwnTaskExperts(base).eval()
    cold.import_export(payload, expected_sources=sources)
    with torch.no_grad():
        if any(not torch.equal(model(ids, expert=name)['logits'], cold(ids, expert=name)['logits']) for name in EXPERTS):
            raise ValueError('Cold task-parameter reload must reproduce every selected computation')
    for path, digest in bindings.items():
        if file_hash(path) != digest:
            raise ValueError('Proof source changed')
    atomic_json(REPORTS / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(),
        actual_counts=model.counts(), actual_selected_Adam_states=len(optimizer.state),
        actual_selected_clock=1, actual_own_complete_train_position=position,
        actual_context_positions=len(ids_), actual_response_EOS_targets=sum(x != -100 for x in labels_),
        actual_disposable_full_model_actor_updates=1, no_production_model_or_live_source_changed=True,
        selected_gradient_norm_before_clip=norm, online_example_loss_before=base_loss, online_example_loss_after=after_loss,
        online_example_loss_not_generalization=True, base_value_versions_pointers_flags_unchanged=True,
        other_expert_values_and_actual_logits_unchanged=True, all_experts_cold_reload_logits_exact=True,
        every_native_view_counts_entire_graph=True, exported_adapter_binding=sources,
        frozen_dense_sha256=base_before, automatic_router_untrained_explicit_expert_required=True,
        vision_encoder_shared_frozen_no_new_image_learning=True,
        public_held_code_or_broad_capability_gain_tested=False, candidate_promoted=False,
        seconds=time.monotonic() - started, source_files_sha256=bindings, goal_achieved=False))
    print(json.dumps(dict(counts=model.counts(), before=base_loss, after=after_loss,
        seconds=time.monotonic() - started, capability_gain_tested=False)), flush=True)


if __name__ == '__main__':
    main()
