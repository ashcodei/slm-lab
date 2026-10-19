"""Validate exact public/retained source positions and Adam clocks before restore."""
from collections import Counter
import math

from .fast_joint_checkpoint_v2 import bundle, restore
from .fast_joint_stage_v1 import learning_factor
from .public_pretraining_update_v1 import FORMAT as OBJECTIVE

FORMAT = 'own-complete-public-pretraining-all-Adam-and-worker-RNG-v1'


def validate(core, plan, schedule, data):
    metrics = core.get('metrics');completed = core.get('completed')
    if (type(completed) is not int or not 0 < completed <= len(schedule)
            or not isinstance(metrics,list) or len(metrics) != completed):
        raise ValueError('Every actual update of a complete original source prefix is required')
    clocks = Counter()
    options = plan['training_options']
    for i, row in enumerate(metrics):
        name, positions = schedule.at(i);kind = data.kinds[name]
        groups = ['language'] if kind == 'text' else ['vision']+(['word'] if kind == 'word' else [])
        factor = learning_factor(i, len(schedule), options['warmup_fraction'], options['cosine_floor'])
        rates = {n:v*factor for n,v in options['learning_rates'].items()}
        if (row.get('step') != i+1 or row.get('objective_format') != OBJECTIVE
                or row.get('family') != name or row.get('global_original_positions') != positions
                or row.get('global_examples') != len(positions)
                or row.get('complete_context_tokens') != sum(data.lengths[name][p] for p in positions)
                or row.get('response_EOS_targets') != sum(data.targets[name][p] for p in positions)
                or row.get('words') != (len(positions) if kind == 'word' else 0)
                or row.get('actual_active_groups') != groups or row.get('reference_forward_queried') is not False
                or row.get('resident_training_only_reference_parameters') != 0 or row.get('learning_rates') != rates
                or not all(isinstance(row.get(k),(int,float)) and math.isfinite(row[k]) for k in ['CE_sum','CTC_word_sum','gradient_norm_before_clip'])):
            raise ValueError('Every actual source, mask denominator, modality and learning rate must reconstruct before mutation')
        clocks.update(groups)
    optimizer = core['optimizer']
    declared = {key for group in optimizer['param_groups'] for key in group['params']}
    if set(optimizer['state'])-declared:raise ValueError('No undeclared optimizer state may enter recovery')
    if {g['name'] for g in optimizer['param_groups']} != {'language','vision','word'}:
        raise ValueError('Every original full-model optimizer partition required')
    for group in optimizer['param_groups']:
        for key in group['params']:
            saved = optimizer['state'].get(key);steps = clocks[group['name']]
            if (steps == 0 and saved is not None
                    or steps > 0 and (saved is None or float(saved['step']) != steps)):
                raise ValueError('Every and only active full-model leaf must retain its exact actual Adam clock')


def make_state(model, optimizer, plan, digest, completed, metrics, rngs, mode, *, schedule, data):
    core = bundle(model, optimizer, plan, digest, completed, metrics, rngs, mode)
    validate(core, plan, schedule, data)
    return dict(format=FORMAT, schedule_identity_sha256=schedule.identity_sha256,
        selected_source_budget=schedule.selected_budget(data), core=core, goal_achieved=False)


def restore_state(model, optimizer, state, plan, digest, device, rank, world, mode, *, schedule, data):
    if (state.get('format') != FORMAT or state.get('schedule_identity_sha256') != schedule.identity_sha256
            or state.get('selected_source_budget') != schedule.selected_budget(data) or state.get('goal_achieved') is not False):
        raise ValueError('Exact complete public/retained source schedule and objective required')
    core = state.get('core', {})
    validate(core, plan, schedule, data)
    return restore(model, optimizer, core, plan, digest, device, rank, world, mode)
