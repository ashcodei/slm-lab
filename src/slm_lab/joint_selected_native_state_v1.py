"""Rebuild a disposable CPU joint actor from one complete selected checkpoint.

Read every original source metric and sparse Adam clock before copying any
selected weight. This is export reconstruction, not training recovery: no
optimizer or private generator is restored, and CUDA generator semantics are
not replayed by this CPU process. Actual GPU admission stays with its owner.
"""
from dataclasses import asdict

import torch

from .codenet_checkpoint_v1 import capture_rng
from .closed_source_recovery_guard_v1 import SourceOnlyRecoveryGuard
from .code_residual_checkpoint_v1 import _validate_rng
from .frozen_own_text_reference_v1 import FORMAT as TEXT_REFERENCE
from .joint_indexed_checkpoint_v3 import configuration, scan, checked_owner
from .joint_indexed_durable_state_v3 import read_latest
from .parallel_function_repair_stage_v1 import value_digest
from .shared_joint_checkpoint_v2 import FixedGuard, SOURCE_KEYS, common_digest, selected, sha, validate_adam
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-complete-joint-selected-state-to-disposable-CPU-native-reconstruction-v1'
OBJECTIVE_FIELDS = {'task', 'source_family', 'source_role', 'source_sha256', 'original_positions',
    'complete_context_tokens', 'response_EOS_targets', 'word_UTF8_bytes', 'normalization_denominators'}


def source_entries(view, sources, entries):
    if (type(sources) is not dict or set(sources) != SOURCE_KEYS or not all(sha(v) for v in sources.values())
            or type(entries) is not list or not entries or sources['complete_schedule_sha256'] != value_digest(entries)):
        raise ValueError('The exact entire original joint schedule and all source identities are required')
    for step, row in enumerate(entries, 1):
        if type(row) is not dict or set(row) != {'step', 'objectives'} or type(row['step']) is not int or row['step'] != step or not row['objectives']:
            raise ValueError('Every full original joint source step must remain in order')
        for objective in row['objectives']:
            if (type(objective) is not dict or set(objective) != OBJECTIVE_FIELDS
                    or objective['task'] not in ['text', 'native_image', 'image_matching', 'word_CTC']
                    or objective['source_role'] != 'train' or objective['source_sha256'] != sources['source_admission_sha256']
                    or type(objective['source_family']) is not str or not objective['source_family']
                    or type(objective['original_positions']) is not list or not objective['original_positions']
                    or len(set(objective['original_positions'])) != len(objective['original_positions'])
                    or any(type(p) is not int or p < 0 for p in objective['original_positions'])
                    or any(type(objective[k]) is not int or objective[k] < 0 for k in ['complete_context_tokens', 'response_EOS_targets', 'word_UTF8_bytes'])
                    or type(objective['normalization_denominators']) is not dict or not objective['normalization_denominators']
                    or any(type(k) is not str or not k or type(v) not in [int, float] or not 0 < v < float('inf')
                        for k, v in objective['normalization_denominators'].items())):
                raise ValueError('Every original train source, target, independent normalizer and active task must remain')
            if objective['task'] == 'word_CTC':
                valid = objective['word_UTF8_bytes'] > 0 and objective['response_EOS_targets'] == objective['complete_context_tokens'] == 0
            else:
                valid = 0 < objective['response_EOS_targets'] <= objective['complete_context_tokens'] and objective['word_UTF8_bytes'] == 0
            if not valid or not view.task_names(objective['task']):
                raise ValueError('Whole native response/EOS targets and CTC bytes must remain independent')


def install(view, optimizer, directory, *, sources, entries, source_files, expected_world, expected_completed,
            expected_pointer_sha256, mode, allow_mechanics=False):
    named = selected(view, optimizer); source_entries(view, sources, entries)
    counts = view.actor.counts(); full = counts['actual_unique'] == 999706442 and counts['frozen_dense_and_all_heads'] == 984399690
    if (type(expected_world) is not int or not 1 <= expected_world <= 4 or type(expected_completed) is not int
            or expected_completed != len(entries) or not sha(expected_pointer_sha256) or optimizer.state
            or mode not in ['capacity', 'train', 'mechanics']
            or mode == 'train' and (expected_completed != 5362 or not full)
            or mode == 'capacity' and expected_completed != 47
            or not full and not (allow_mechanics is True and counts['actual_unique'] < 10**7)
            or full and (expected_world != 4 or mode == 'mechanics')
            or any(p.device.type != 'cpu' or p.dtype != torch.float32 or p.grad is not None for p in view.parameters())
            or view.training or torch.cuda.is_initialized()):
        raise ValueError('Complete source-bound CPU export reconstruction requires its exact world,full cursor,initial graph and mode')
    fixed = FixedGuard(view, optimizer)
    if (fixed.sha256 != sources['fixed_values_sha256']
            or tensor_digest(view.actor.base.language_named_parameters()) != sources['text_initializer_sha256']
            or tensor_digest((n, p) for n, p in view.actor.base.named_parameters()
                if n.startswith(('network.vision.', 'heads.pointing.', 'heads.word.'))) != sources['visual_initializer_sha256']):
        raise ValueError('The complete exact original own native initializer and fixed values must precede every selected copy')
    before = dict(parameters=tensor_digest(view.named_parameters()), Adam=value_digest(optimizer.state_dict()), RNG=value_digest(capture_rng('cpu')))
    shared, receipt, pointer_sha256 = read_latest(directory, expected_pointer_sha256=expected_pointer_sha256)
    state = shared['common']; history = state['history_configuration']
    binding_identity = value_digest(dict(sources=sources, entries=entries, source_files=source_files))

    def resource_guard():
        fixed.check(view, optimizer)
        if value_digest(dict(sources=sources, entries=entries, source_files=source_files)) != binding_identity:
            raise ValueError('All original export source files,schedule and model metadata must remain unchanged')

    checked = SourceOnlyRecoveryGuard(source_files, checked_owner(resource_guard))
    configuration(history, sources=sources, metadata=fixed.metadata, world=expected_world, optimizer=optimizer, planned=len(entries))
    reference = history['binding']['own_text_reference']
    if (reference.get('format') != TEXT_REFERENCE or reference.get('configuration') != asdict(view.actor.config)
            or reference.get('frozen_language_sha256') != sources['text_initializer_sha256']
            or reference.get('tokenizer_sha256') != sources['tokenizer_sha256'] or reference.get('reference_is_text_only') is not True
            or reference.get('retained_extra_neural_parameters') != 0):
        raise ValueError('The actual immutable own text reference must match the original shared initializer')
    cursor = scan(view, entries, expected_completed, state['history_receipts'], history, checked.check)
    if (state['sources'] != sources or state['world_size'] != expected_world or state['model_metadata'] != fixed.metadata
            or state['cursor'] != cursor or receipt['completed'] != expected_completed
            or list(state['selected_weights']) != [n for n, _ in named] or state['optimized_names'] != [n for n, _ in named]
            or common_digest(state) != receipt['semantic_manifest']['actual_common_sha256']):
        raise ValueError('The complete selected state,original source prefix,external receipt and exact parameter inventory must agree')
    for name, parameter in named:
        value = state['selected_weights'][name]
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.dtype != parameter.dtype
                or value.shape != parameter.shape or not bool(torch.isfinite(value).all())):
            raise ValueError('Every finite exact-shape joint selected tensor must validate before any model copy')
    validate_adam(state['optimizer'], optimizer, named, cursor['actual_per_parameter_Adam_steps'])
    for private in shared['private_states']:
        rng = private['private_RNG']; cpu = dict(rng, cuda=None)
        _validate_rng(cpu, 'cpu')
        if rng['cuda'] is not None and (not isinstance(rng['cuda'], torch.Tensor) or rng['cuda'].device.type != 'cpu'
                or rng['cuda'].dtype != torch.uint8 or rng['cuda'].ndim != 1 or rng['cuda'].numel() < 1):
            raise ValueError('Every retained CUDA private generator must retain its complete CPU byte payload')
        if full and rng['cuda'] is None:
            raise ValueError('The full-size GPUproducer must retain every actual CUDA generator')
    checked.verify_final(); resource_guard()
    if (before != dict(parameters=tensor_digest(view.named_parameters()), Adam=value_digest(optimizer.state_dict()), RNG=value_digest(capture_rng('cpu')))
            or pointer_sha256 != expected_pointer_sha256):
        raise ValueError('Complete source/state validation must finish before changing any disposable model value')
    with torch.no_grad():
        for name, parameter in named:
            parameter.copy_(state['selected_weights'][name])
    fixed.check(view, optimizer, full=True)
    if (tensor_digest(view.named_selected()) != tensor_digest(state['selected_weights'].items()) or optimizer.state
            or value_digest(capture_rng('cpu')) != before['RNG'] or torch.cuda.is_initialized()):
        raise ValueError('Only the completed selected weights may change;the export process cannot resume Adam,RNG or CUDA')
    return dict(format=FORMAT, mode=mode, completed_updates=expected_completed, original_world=expected_world,
        actual_counts=counts, sources=dict(sources), actual_checkpoint_sha256=receipt['checkpoint_sha256'],
        actual_original_pointer_sha256=pointer_sha256, actual_complete_cursor=cursor,
        actual_selected_weights_sha256=tensor_digest(view.named_selected()), actual_complete_Adam_sha256=value_digest(state['optimizer']),
        actual_Adam_states_validated=len(state['optimizer']['state']), original_all_private_RNG_sha256=receipt['semantic_manifest']['all_actual_private_RNG_sha256'],
        every_original_metric_and_sparse_Adam_clock_reconstructed=True, complete_validation_before_selected_copy=True,
        original_fixed_values_and_current_CPU_RNG_unchanged=True, optimizer_or_private_generators_not_restored=True,
        CUDA_generator_semantics_not_replayed_by_CPU_exporter=True, new_optimizer_updates=0,
        full_size_GPU_capacity_or_training_admission=False, candidate_promoted=False, goal_achieved=False)
