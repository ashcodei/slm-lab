"""Full native code CE plus execution-labelled length-normalized preference.

Inspired by SimPO (https://arxiv.org/abs/2405.14734). Our extension uses
execution labels on our own unassisted deliveries, including explicitly
censored budget-exhausted negatives, and an own-starting-policy KL anchor.
This is CPU-only experimental machinery, not human RLHF or GPU admission.
"""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .codenet_checkpoint_v1 import FileBindings
from .own_dense_actor_v1 import OwnDenseActor


def contrastive_loss(chosen_mean_logp, rejected_mean_logp, *, beta=2., margin=.5):
    if (chosen_mean_logp.shape != rejected_mean_logp.shape or not chosen_mean_logp.numel()
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in [beta, margin])
            or beta <= 0 or margin < 0 or not bool(torch.isfinite(chosen_mean_logp).all())
            or not bool(torch.isfinite(rejected_mean_logp).all())):
        raise ValueError('Finite matching mean action log probabilities and explicit positive beta/nonnegative margin required')
    return F.softplus(margin - beta * (chosen_mean_logp - rejected_mean_logp)).mean()


class ExecutionContrastPairs:
    role = 'train'
    purpose = 'original_code_CE_and_own_execution_preferences'

    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.manifest = json.loads((self.folder / 'manifest.json').read_text())
        meta = self.manifest
        if (meta.get('format') != 'own-original-human-code-versus-execution-rejected-unassisted-CPU-pairs-v1'
                or meta.get('pairs') != 20 or meta.get('distinct_original_train_sources') != 4
                or meta.get('human_rankings') is not False or meta.get('held_assessment') is not False
                or meta.get('external_teacher') is not False
                or not meta.get('all_original_reference_and_actual_wrong_deliveries_freshly_reexecuted')):
            raise ValueError('Exact complete original20-pair execution-labelled train admission required')
        bindings = dict(meta['source_files_sha256'])
        bindings.update({str(self.folder / name): value for name, value in meta['files_sha256'].items()})
        self.guard = FileBindings(bindings); self.guard.check(full=True)
        self.rows = [json.loads(line) for line in (self.folder / 'pairs.jsonl').read_text().splitlines()]
        if len(self.rows) != 20 or len({r['pair_id'] for r in self.rows}) != 20:
            raise ValueError('Every original pair exactly once required')
        self.by_source = {}
        for index, row in enumerate(self.rows):
            validate_pair(row)
            if row['pair_index'] != index: raise ValueError('Exact source order required')
            key = row['source_id']
            if key in self.by_source:
                first = self.rows[self.by_source[key]]
                if any(row[k] != first[k] for k in ['source_position', 'complete_chosen_ids', 'complete_chosen_labels', 'original_native_prefix_ids']):
                    raise ValueError('One original positive per source must remain unchanged across negatives')
            else: self.by_source[key] = index
        if (len(self.by_source) != 4 or sum(r['rejected_generated_EOS'] for r in self.rows) != meta['genuine_EOS_negatives']
                or sum(len(r['rejected_actual_action_ids']) for r in self.rows) != meta['negative_actual_action_targets']):
            raise ValueError('All original source/delivery/target accounting must reconstruct')


def validate_pair(row):
    positive = row['complete_chosen_ids']; negative = row['complete_rejected_ids']; prefix = row['original_native_prefix_ids']
    actions = row['rejected_actual_action_ids']; start = len(prefix)
    if (row.get('original_split') != 'train' or row.get('publisher_split') != 'train'
            or row.get('machine_execution_preference_not_human_ranked') is not True
            or row.get('external_model_teacher') is not False or row.get('positive_finite_case_credit') != 1.
            or row.get('negative_finite_case_credit') != 0. or not prefix or prefix[0] != 1
            or positive[:start] != prefix or negative != prefix + actions or not actions
            or positive[-1] != 2 or row['complete_chosen_labels'] != [-100] * start + positive[start:]
            or row['complete_rejected_labels'] != [-100] * start + actions
            or len(positive) - start != row['complete_original_reference_EOS_targets']
            or any(type(t) is not int or not 0 <= t < 32768 for t in positive + negative)
            or any(t in (0, 1, 2) for t in positive[start:-1]) or 2 in actions[:-1]
            or len(positive) > 4096 or len(negative) > 4096
            or row['rejected_generated_EOS'] != (actions[-1] == 2)
            or row['rejected_is_censored_at_complete512_action_budget'] != (actions[-1] != 2)
            or (actions[-1] != 2 and len(actions) != 512)
            or row.get('rejected_full_EOS_was_not_invented') is not True):
        raise ValueError('Complete native train positive and exact actual negative boundary/labels required')


def batch(row, which):
    if which not in ['chosen', 'rejected']: raise ValueError('Explicit complete original pair side required')
    return dict(input_ids=torch.tensor([row['complete_' + which + '_ids']], dtype=torch.long),
                labels=torch.tensor([row['complete_' + which + '_labels']], dtype=torch.long))


def update(model, optimizer, row, own_initial_logps, *, method, beta=2., margin=.5, preference_weight=.2, KL_weight=.01):
    if (type(model) is not OwnDenseActor or next(model.parameters()).device.type != 'cpu'
            or any(p.dtype != torch.float32 for p in model.parameters()) or method not in ['anchor_CE', 'anchor_contrast']
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in [preference_weight, KL_weight])):
        raise ValueError('Exact own CPU FP32 actor and declared anchored CE/contrastive method required')
    validate_pair(row); model.assert_shared(); model.eval()
    selected = [p for _, p in model.named_selected(('actor',))]; chosen_ids = {id(p) for p in selected}
    if [id(p) for g in optimizer.param_groups for p in g['params']] != [id(p) for p in selected]:
        raise ValueError('Optimizer must contain exactly the original actor parameters in original order')
    fixed = [p for p in model.parameters() if id(p) not in chosen_ids]
    signature = lambda: [(id(p), p._version, p.data_ptr(), p.requires_grad) for p in fixed]
    before = signature()
    if any(p.grad is not None for p in model.parameters()): raise ValueError('Every gradient must start cleared')
    preferred = batch(row, 'chosen'); rejected = batch(row, 'rejected')
    expected = (1, preferred['input_ids'].shape[1] - 1, model.config.vocab_size)
    if (own_initial_logps.shape != expected or own_initial_logps.dtype != torch.float32
            or own_initial_logps.device.type != 'cpu' or own_initial_logps.requires_grad
            or not bool(torch.isfinite(own_initial_logps).all())
            or float(own_initial_logps.max()) > 1e-5
            or float(own_initial_logps.logsumexp(-1).abs().max()) > 1e-4):
        raise ValueError('Complete normalized own initial next-token distributions required')
    positive_logits = model(preferred['input_ids'])['logits'][:, :-1].float()
    positive_logps = positive_logits.log_softmax(-1)
    mask = preferred['labels'][:, 1:] != -100
    chosen_mean = positive_logps.gather(-1, preferred['input_ids'][:, 1:, None]).squeeze(-1)[mask].mean()
    # The control observes the same negative but gives it no training gradient.
    with torch.enable_grad() if method == 'anchor_contrast' else torch.no_grad():
        negative_logits = model(rejected['input_ids'])['logits'][:, :-1].float()
        negative_mask = rejected['labels'][:, 1:] != -100
        rejected_mean = negative_logits.log_softmax(-1).gather(-1, rejected['input_ids'][:, 1:, None]).squeeze(-1)[negative_mask].mean()
    preference = contrastive_loss(chosen_mean, rejected_mean, beta=beta, margin=margin)
    reference_KL = (own_initial_logps.exp() * (own_initial_logps - positive_logps)).sum(-1).mean()
    anchor = -chosen_mean
    loss = anchor + KL_weight * reference_KL + (preference_weight * preference if method == 'anchor_contrast' else 0.)
    if not bool(torch.isfinite(loss)): raise ValueError('Finite whole native objective required before backward')
    loss.backward(inputs=selected)
    if (signature() != before or any(p.grad is not None for p in fixed)
            or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in selected)):
        raise ValueError('Only selected original actor parameters may receive finite gradients')
    grad = float(torch.nn.utils.clip_grad_norm_(selected, 1., error_if_nonfinite=True))
    record = dict(method=method, pair_index=row['pair_index'], pair_id=row['pair_id'], source_id=row['source_id'],
        original_reference_EOS_targets=int(mask.sum()), genuine_negative_action_targets=int(negative_mask.sum()),
        original_reference_CE=float(anchor.detach()), mean_chosen_logp=float(chosen_mean.detach()),
        mean_rejected_logp=float(rejected_mean.detach()), length_normalized_preference_loss=float(preference.detach()),
        own_initial_full_context_forward_KL=float(reference_KL.detach()), actual_objective=float(loss.detach()),
        gradient_norm_before_clip=grad, beta=beta, margin=margin, chosen_CE_weight=1.,
        preference_weight=preference_weight if method == 'anchor_contrast' else 0., KL_weight=KL_weight,
        rejected_actual_EOS=row['rejected_generated_EOS'], censored_negative_boundary_not_completed=row['rejected_is_censored_at_complete512_action_budget'],
        all_original_reference_EOS_targets_preserved=True, no_human_preference_or_GPU_admission=True,
        native_pair_sha256=hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    if signature() != before: raise ValueError('Frozen backbone/vision/heads/critic changed during actor update')
    return record
