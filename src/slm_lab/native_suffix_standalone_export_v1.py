"""Fold complete own native-code training into one standalone multimodal model.

The original source schedule and CE masks remain required. Folding changes
floating-point arithmetic order; cold equality does not establish equivalent
hot-policy generations or model quality. The dense critic and all original
image/word/auxiliary heads are retained; the unused research critic is omitted.
"""
import copy

import torch

from .deterministic_positions_v1 import vision_forward, patch_grid
from .corpus import file_hash
from .native_suffix_train_data_v1 import NativeSuffixTrainData
from .native_suffix_training_controller_v1 import TRAINING
from .own_dense_actor_fold_v1 import fold_actor
from .own_dense_actor_v1 import OwnDenseActor
from .own_dense_standalone_v1 import export as export_dense
from .parallel_native_suffix_stage_v1 import FORMAT as STATE, CE_ONLY, KERNEL
from .parallel_function_repair_stage_v1 import value_digest
from .whole_human_sft_evidence_v1 import tensor_digest

PROFILE = 'own-complete-native-code-suffix-standalone-multimodal-candidate-v1'
FULL = dict(updates=11172, source_presentations=44652, distinct_original_human_sources=3721,
    complete_original_context_tokens=27450264, original_response_EOS_targets=7010616,
    actual_single_source_three_empty_rank_steps=12)


def export(model, data, state, receipt, *, sources, tokenizer_sha256, working_budget_bytes, allow_mechanics=False):
    if (type(model) is not OwnDenseActor or type(data) is not NativeSuffixTrainData or data.role != 'train'
            or receipt.get('format') != TRAINING or receipt.get('complete') is not True
            or state.get('format') != STATE or state.get('mode') != receipt.get('mode')
            or state.get('profile') != receipt.get('profile') or receipt.get('profile') not in ['anchor_CE', 'anchor_execution']
            or state.get('world_size') != 4 or receipt.get('world') != 4 or state.get('groups') != ['actor']
            or state.get('completed') != state.get('planned') or state.get('completed') != receipt.get('actual_complete_updates')
            or len(state.get('metrics', [])) != state.get('completed')
            or receipt.get('actual_research_actor_parameters') != model.counts()['actual_unique']
            or receipt.get('actual_native_parameters') != model.base.counts()['total_unique']
            or receipt.get('every_original_complete_source_and_CE_target_retained') is not True
            or receipt.get('candidate_promoted') is not False or receipt.get('goal_achieved') is not False):
        raise ValueError('The exact complete native actor-only training state, source and controller receipt are required')
    if (getattr(model.base.network.vision.forward, '__func__', None) is not vision_forward
            or getattr(model.base.patch_grid, '__func__', None) is not patch_grid):
        raise ValueError('The canonical deterministic native vision/word position protocol is required')
    data.guard.check()
    if tokenizer_sha256 != file_hash(data.corpus.tokenizer_path):
        raise ValueError('The exact original own source tokenizer must be packaged')
    mode = state['mode']
    entries = state['global_cursor']['all_consumed_global_source_batches']
    budget = data.budget(entries)
    if budget != receipt['original_complete_training_budget'] or state['schedule_identity_sha256'] != data.identity(entries):
        raise ValueError('Every actual original source, full CE/EOS target and declared source cursor must reconstruct')
    if mode == 'train':
        if (data.purpose != 'optimization' or not data.full_source_data_admitted or len(data) != 14884
                or not data.admission_sha256 or not data.manifest_sha256
                or entries != data.complete_schedule(epochs=3, width=4, seed=989711)
                or any(budget.get(k) != v for k, v in FULL.items())
                or receipt.get('production_updates') != 11172
                or model.counts()['actual_unique'] != 999706442 or model.base.counts()['total_unique'] != 984399690):
            raise ValueError('Full native export requires all3721sources/14884views/44652presentations/7010616targets/11172updates')
        if any(sources.get(name) != digest for name, digest in data.source_files_sha256.items()):
            raise ValueError('The entire actual native data admission and every exact original input must remain bound')
    elif (mode != 'mechanics' or allow_mechanics is not True or data.purpose != 'CPU-mechanics'
            or data.full_source_data_admitted or model.counts()['actual_unique'] >= 10**6
            or receipt.get('production_updates') != 0 or torch.cuda.is_initialized()):
        raise ValueError('Only explicit own-small CPU mechanics can export without complete production scope')
    for index, (positions, row) in enumerate(zip(entries, state['metrics'])):
        infos = [data.information(p) for p in positions]
        if (row['step'] != index + 1 or row['actual_original_source_positions'] != positions
                or row['global_original_sources'] != infos or row['actual_world_size'] != 4
                or row['original_response_EOS_targets'] != sum(i['original_response_EOS_targets'] for i in infos)
                or row['complete_original_context_tokens'] != sum(i['original_context_tokens'] for i in infos)
                or len(row['actual_rank_observations']) != 4):
            raise ValueError('Every declared whole source update and complete original CE mask must be retained')
        for rank, observed in enumerate(row['actual_rank_observations']):
            local = positions[rank:rank+1]
            kernel = observed['kernel']
            if observed['rank'] != rank or observed['actual_original_positions'] != local:
                raise ValueError('Actual original rank source identities cannot change during export')
            if not local:
                if kernel is not None: raise ValueError('Empty ranks cannot acquire a source during export')
            elif (kernel['format'] != (KERNEL if infos[rank]['native_execution_eligible'] else CE_ONLY)
                    or kernel['unconditional_original_CE_anchor']['local_original_response_EOS_targets'] != infos[rank]['original_response_EOS_targets']
                    or kernel['own_reference_binding'] != state['reference_binding']):
                raise ValueError('Every native qualification and complete unconditional CE fallback must remain')
    model.validate_export(state['model'], expected_sources=sources)
    if (tensor_digest(model.parts.named_parameters()) != receipt['actual_final_actor_sha256']
            or tensor_digest(state['model']['parts'].items()) != receipt['actual_final_actor_sha256']
            or any(not torch.equal(value.detach().cpu(), state['model']['parts'][name]) for name, value in model.parts.state_dict().items())
            or value_digest(state['optimizer']) != receipt['actual_final_Adam_sha256']
            or any(not torch.equal(p, torch.zeros_like(p)) for p in model.parts['critic'].parameters())):
        raise ValueError('All actual completed actor/Adam values and the unused zero research critic must match')
    data.guard.check(full=True)
    native, fold = fold_actor(model, sources=sources, working_budget_bytes=working_budget_bytes)
    with torch.no_grad(): native.network.value_head.weight.copy_(model.base.network.value_head.weight)
    if not torch.equal(native.network.value_head.weight, model.base.network.value_head.weight):
        raise ValueError('The original dense critic must survive actor-only native coding continuation')
    provenance = dict(profile=PROFILE, mode=mode, training_profile=state['profile'], complete_training_budget=copy.deepcopy(budget),
        actual_completed_updates=state['completed'], whole_production_source_schedule_completed=mode == 'train',
        actual_native_data_admission_sha256=data.admission_sha256, actual_native_data_manifest_sha256=data.manifest_sha256,
        completed_training_actor_sha256=receipt['actual_final_actor_sha256'], completed_training_Adam_sha256=receipt['actual_final_Adam_sha256'],
        original_dense_critic_preserved=True, unused_zero_research_critic_excluded=True,
        every_original_native_source_CE_mask_and_failed_disposition_retained=True,
        final_native_parameters_sha256=tensor_digest(native.named_parameters()), original_nonmutating_fold_observation=fold,
        actor_to_folded_arithmetic_order_changes=True, hot_policy_FP32_BF16_or_generation_equivalence_not_assumed=True,
        no_external_pretrained_model_weights_or_live_teacher_API=True,
        actual_full_candidate_native_cold_and_all_original_quality_assessments_still_required=True)
    payload = export_dense(native, sources=sources, provenance=provenance, completed_updates=state['completed'], tokenizer_sha256=tokenizer_sha256)
    return native, payload, provenance
