"""Fixed-action full native shape stress with backward and no optimizer commit.

Forced source-token streams are capacity controls, not sampled RL trajectories.
Cached/full probabilities are still actual evaluations of those same tokens.
"""
import torch

from .anchored_code_objective_v1 import reference_anchor
from .codenet_checkpoint_v1 import capture_rng
from .frozen_own_actor_reference_v1 import signature
from .whole_human_sft_evidence_v1 import tensor_digest


def equal(a, b):
    if isinstance(a, torch.Tensor): return torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def adam_digest(optimizer):
    return tensor_digest((str(i)+'.'+key, value) for i, state in optimizer.state_dict()['state'].items()
                         for key, value in state.items())


def stress(model, optimizer, corpus, own_reference, position, *, cache_absolute_bound):
    row = corpus.row(position); sequence, labels = corpus.sequence(position)
    start = row['prefix_positions']; prefix = sequence[:start]; horizon = row['full_training_action_budget']
    if len(prefix) + horizon > model.config.max_text_length:
        raise ValueError('Full original prefix and complete declared action horizon must fit')
    content = [token for token in sequence[start:-1] if token not in (0, 1, 2)]
    if not content: raise ValueError('Actual original non-special source tokens required')
    actions = (content * ((horizon + len(content) - 1) // len(content)))[:horizon]
    if len(actions) != horizon or any(token in (0, 1, 2) for token in actions):
        raise ValueError('Full forced non-EOS action stream required')
    device = next(model.parameters()).device
    parameters = [p for _, p in model.named_selected(('actor',))]
    fixed = [p for p in model.parameters() if id(p) not in {id(q) for q in parameters}]
    versions = signature(model.parameters()); frozen_reference = own_reference.binding()
    moments = adam_digest(optimizer); rng = capture_rng(device)
    own_reference.assert_bound(model); model.eval(); cached = []
    try:
        with torch.no_grad(), torch.autocast(device.type, enabled=False):
            state = model.policy.prefill(torch.tensor([prefix], device=device))
            for i, token in enumerate(actions):
                logits = state['logits'][0, -1].float()
                cached.append(float(logits.log_softmax(-1)[token]))
                if i + 1 < len(actions):
                    state = model.policy.decode(torch.tensor([[token]], device=device), state['cache'])
            del state
        model.train(); model.policy.gradient_checkpointing = True; optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=False):
            ids = torch.tensor([prefix + actions], device=device)
            logits = model(ids)['logits'][0, len(prefix)-1:-1]
            with torch.no_grad(): reference = own_reference(ids)['logits'][0, len(prefix)-1:-1]
            logp = logits.float().log_softmax(-1); logq = reference.detach().float().log_softmax(-1)
            selected = logp.gather(1, torch.tensor(actions, device=device)[:, None]).flatten()
            maximum = float((selected.detach() - torch.tensor(cached, device=device)).abs().max())
            if maximum > cache_absolute_bound:
                raise ValueError('Full forced-horizon actual cached/full rounding exceeds runtime guard: ' + str(maximum))
            # This is a declared synthetic NLL/KL capacity objective. It is
            # never described as an on-policy reward gradient or committed.
            loss = -selected.mean() + .01 * (logq.exp() * (logq - logp)).sum() / horizon
            loss.backward(inputs=parameters)
            synthetic_loss = float(loss.detach()); del logits, reference, logp, logq, selected, loss
            batch = corpus.batch(position, device); targets = batch.pop('labels')
            logits = model(**batch)['logits']
            anchor, metric = reference_anchor(logits, targets, row['response_EOS_targets'])
            anchor.backward(inputs=parameters); del logits, anchor
        if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters)
                or any(p.grad is not None for p in fixed)):
            raise ValueError('Actual full-shape backward requires every selected and no fixed gradient')
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
    finally:
        optimizer.zero_grad(set_to_none=True)
    own_reference.assert_bound(model)
    if (versions != signature(model.parameters()) or moments != adam_digest(optimizer)
            or frozen_reference != own_reference.binding() or not equal(rng, capture_rng(device))):
        raise ValueError('Capacity-only forward/backward changed values, Adam, reference or private RNG')
    return dict(original_source_id=row['id'], original_source_position=position,
        original_complete_CE_context_positions=len(sequence), original_reference_EOS_targets=row['response_EOS_targets'],
        actual_complete_prefix_positions=len(prefix), actual_forced_action_tokens=horizon,
        actual_full_action_context_positions=len(prefix) + horizon,
        non_EOS_actions_are_repeated_original_source_tokens=True, genuine_sampled_RL_trajectory=False,
        actual_full_cached_action_logps=cached, actual_cache_full_max_absolute_difference=maximum,
        synthetic_capacity_NLL_KL_loss=synthetic_loss, complete_original_reference_CE_sum=metric['reference_CE_sum'],
        gradient_norm_before_clip=norm, actual_selected_gradient_tensors=len(parameters),
        actual_values_Adam_reference_all_private_RNG_unchanged=True, optimizer_commits=0,
        useful_code_or_RL_improvement_proven=False, goal_achieved=False)
