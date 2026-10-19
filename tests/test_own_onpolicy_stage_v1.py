from contextlib import nullcontext
import copy

import pytest
import torch
from torch.nn import functional as F

from slm_lab.execution_preference_v1 import reference_scores
from slm_lab.own_code_training_stage_v1 import make_optimizer
from slm_lab.own_onpolicy_stage_v1 import OwnOnPolicyStage
from test_own_code_training_stage_v1 import fixtures as prior_fixtures


def fixtures(alpha=.1):
    model, previous, opts = prior_fixtures()
    streams = {k: previous[k] for k in ['generation', 'variety']}
    options = {**opts, 'world_size': 1, 'microbatch_per_rank': 1, 'accumulation': 1,
        'global_examples_per_stream': 1, 'steps': 5, 'coefficients': dict(generation=1., variety=.25),
        'DPO_alpha': alpha, 'DPO_beta': .1}
    pairs = []
    for i in range(4):
        prefix = [1, i+4]; chosen = [8+i, 9+i, 2]; rejected = [20+i, 2]
        pairs.append(dict(id='actual-fixture-'+str(i), split='train', publisher_split='train',
            chosen=[prefix+chosen, [-100]*len(prefix)+chosen], rejected=[prefix+rejected, [-100]*len(prefix)+rejected],
            feedback_source='machine_execution_of_original_publisher_tests', finite_execution_verified=True,
            human_preferences=False, negative_is_actual_complete_own_policy_response=True,
            original_human_target_bytes_unchanged=True, reference_policy_sha256='a'*64))
    model.eval()
    reference = reference_scores(model.code, pairs, policy_sha256='a'*64, device='cpu', amp=nullcontext)
    sources = dict(initial_policy_manifest_sha256='a'*64, reference_scores_sha256='b'*64, run_plan_sha256='c'*64)
    return model, streams, pairs, reference, options, sources


def test_streamed_ce_and_reference_anchored_DPO_match_explicit_joint_autograd():
    model, streams, pairs, reference, options, sources = fixtures()
    options['gradient_norm_clip'] = 1e6
    direct = copy.deepcopy(model); direct.code.train(); direct.base.eval()
    optimizer = torch.optim.SGD(model.code_parameters(), lr=options['learning_rate'])
    stage = OwnOnPolicyStage(model, optimizer, streams, pairs, reference, options, sources, 'cpu')
    positions = stage.positions(0); pair = pairs[positions['pairs'][0][0]]
    losses = []
    for name in streams:
        batch = streams[name].encoder.batch([dict(packed_code_index=positions[name][0][0])], 'response', 'cpu')
        losses.append(direct.code(**batch)['loss'] * options['coefficients'][name])
    values = []
    for field in ['chosen', 'rejected']:
        ids, labels = pair[field]
        n = sum(v != -100 for v in labels)
        values.append(-n * direct.code(input_ids=torch.tensor([ids]), labels=torch.tensor([labels]))['loss'])
    ref = reference['records'][pair['id']]
    total = sum(losses) + options['DPO_alpha'] * F.softplus(-options['DPO_beta'] *
        (values[0]-values[1]-(ref['chosen_logp']-ref['rejected_logp'])))
    total.backward(); expected = [p.grad.clone() for p in direct.code_parameters()]
    captured = []; original_step = optimizer.step
    def checked_step(*args, **kwargs):
        captured.extend(p.grad.clone() for p in model.code_parameters())
        return original_step(*args, **kwargs)
    optimizer.step = checked_step
    base = {n: t.clone() for n, t in model.base.state_dict().items()}
    record = stage.update()
    assert record['actual_pair_id'] == pair['id']
    assert abs(record['summed_objective'] - float(total.detach())) < 1e-5
    assert all(torch.allclose(a, b, rtol=1e-4, atol=1e-6) for a, b in zip(captured, expected))
    assert all(torch.equal(t, model.base.state_dict()[n]) for n, t in base.items())
    assert all(p.grad is None for p in model.parameters())


def test_reference_component_Adam_source_cursor_and_RNG_snapshot_reproduces_next_update_and_rejects_corruption(tmp_path):
    args = fixtures(); model, streams, pairs, reference, options, sources = args
    stage = OwnOnPolicyStage(model, make_optimizer(model, options), streams, pairs, reference, options, sources, 'cpu')
    base_versions = [(id(p), p._version, p.data_ptr()) for p in model.base.parameters()]
    stage.update(); stage.save(tmp_path/'snapshot.pt', 8*1024**2, 0)
    saved = torch.load(tmp_path/'snapshot.pt', weights_only=True)
    expected = stage.update(); parts = {n: t.clone() for n, t in model.parts.state_dict().items()}
    Adam = copy.deepcopy(stage.optimizer.state_dict()); rng = torch.get_rng_state().clone()
    for mutate in [lambda s: s['sources'].update(reference_scores_sha256='d'*64),
                   lambda s: s['sampler_state']['pairs'].update(actual_exposures=0),
                   lambda s: s['optimizer']['state'][0]['exp_avg'].fill_(float('nan')),
                   lambda s: s['worker_rng_states'][0].update(torch=torch.zeros(1, dtype=torch.uint8)),
                   lambda s: s['options'].update(DPO_alpha=0.)]:
        corrupted = copy.deepcopy(saved); mutate(corrupted)
        with pytest.raises(ValueError):
            stage.resume(corrupted)
        assert stage.completed == 2 and all(torch.equal(t, model.parts.state_dict()[n]) for n, t in parts.items())
        assert torch.equal(rng, torch.get_rng_state())
        assert base_versions == [(id(p), p._version, p.data_ptr()) for p in model.base.parameters()]
    stage.resume(saved); actual = stage.update()
    assert expected == actual
    assert all(torch.equal(t, model.parts.state_dict()[n]) for n, t in parts.items())
    assert torch.equal(rng, torch.get_rng_state())
    for key in Adam['state']:
        for field, tensor in Adam['state'][key].items():
            assert torch.equal(tensor, stage.optimizer.state_dict()['state'][key][field])
    assert base_versions == [(id(p), p._version, p.data_ptr()) for p in model.base.parameters()]


def test_control_has_no_preference_gradient_and_held_or_changed_reference_rejects_before_live_changes():
    model, streams, pairs, reference, options, sources = fixtures(0.)
    stage = OwnOnPolicyStage(model, make_optimizer(model, options), streams, pairs, reference, options, sources, 'cpu')
    record = stage.update(); assert record['DPO']['weighted_DPO'] == 0.
    assert record['DPO']['records'][0]['gradient_coefficient'] == 0.
    current = {n: p.clone() for n, p in model.named_parameters()}
    reference['records'][pairs[0]['id']]['chosen_fingerprint'] = '0'*64
    with pytest.raises(ValueError, match='exact original chosen/rejected'):
        stage.update()
    assert all(torch.equal(current[n], p) for n, p in model.named_parameters())
    streams['generation'].role = 'validation'
    with pytest.raises(ValueError, match='optimization streams'):
        OwnOnPolicyStage(model, None, streams, None, None, options, sources, 'cpu')
