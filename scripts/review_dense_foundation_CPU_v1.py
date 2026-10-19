"""Read both full mixed-CE/KL CPU snapshots and original admitted inputs.

This reconstructs saved tensors, optimizer inventories, source masks and
objective arithmetic. The original probe supplies the actual neural replay;
this reader does not independently repeat that neural computation.
"""
import json
import math
import os
from pathlib import Path

import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.code_residual_checkpoint_v1 import _validate_adam, _validate_rng
from slm_lab.corpus import file_hash
from slm_lab.dense_foundation_data_v1 import DenseFoundationSources, REFERENCE_ROUTES
from slm_lab.dense_foundation_held_v1 import target_digest
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig
from slm_lab.own_dense_growth_v1 import DenseBackbone, OwnDenseAssistant

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/dense-foundation-full-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('The complete CPU snapshot reader must hide every GPU')
    output = REPORTS/'independent-complete-readback.json'
    if output.exists():
        raise ValueError('Preserve previous independent readback')
    torch.set_num_threads(2)
    plan = json.loads((REPORTS/'plan.json').read_text())
    proof = json.loads((REPORTS/'complete.json').read_text())
    guard = FileBindings(plan['source_files_sha256']); guard.check(full=True)
    settings = plan['initial_dense_settings']
    if file_hash(settings['initializer']) != settings['initializer_sha256']:
        raise ValueError('Exact own initial full model required')
    initial = torch.load(settings['initializer'], map_location='cpu', weights_only=True, mmap=True)
    with torch.device('meta'):
        meta = OwnDenseAssistant(DenseBackbone(ModelConfig(**initial['config'])), **initial['settings'])
    named = meta.language_named_parameters(); names = [n for n, _ in named]
    if (meta.counts() != proof['actual_counts'] or meta.counts()['total_unique'] != 984399690
            or sum(p.numel() for _, p in named) != 931244544 or len(names) != 317):
        raise ValueError('Actual full student/head and dense Adam inventory required')
    data = DenseFoundationSources(ROOT, meta.config, plan['CPU_education_window_context'])
    positions = plan['actual_selected_global_train_positions']
    information = [data.information(p) for p in positions]
    if data.populations != plan['source_populations'] or information != plan['complete_selected_original_sources']:
        raise ValueError('All original complete selected source identities must reconstruct')
    source_masks = []
    for position, info in zip(positions, information):
        batch = data.batch(position, 'cpu'); labels = batch['labels'][:, 1:]
        targets = labels[labels != -100].tolist()
        if (len(targets) != info['response_EOS_targets']
                or int(batch['attention_mask'].sum()) != info['complete_context_tokens']):
            raise ValueError('Every original selected context and actual response/next/EOS mask required')
        source_masks.append(dict(**info, original_target_ids_sha256=target_digest(targets)))
    expected_sources = dict(full_CPU_plan_sha256=file_hash(REPORTS/'plan.json'),
        initializer_sha256=settings['initializer_sha256'],
        education_admission_sha256=file_hash(ROOT/'data/open-education-bookpack-v1/admission.json'))
    fixed = [n for n in initial['network'] if n not in names and n != 'lm_head.weight']
    for step in (1, 2):
        path = REPORTS/f'resume-step{step}.pt'
        if file_hash(path) != proof[f'retained_step{step}_sha256']:
            raise ValueError('Both actual complete model/Adam snapshots required')
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        saved = state['model']
        if (state['completed'] != step or state['planned'] != 2 or state['world_size'] != 1
                or state['optimized_names'] != names or state['metrics'] != proof['weighted_complete_source_metrics'][:step]
                or state['cursor'] != dict(original_global_positions=positions, education_window_context=256, completed_updates=step)
                or saved['sources'] != expected_sources or saved['config'] != initial['config']
                or saved['settings'] != initial['settings'] or saved['actual_counts'] != meta.counts()):
            raise ValueError('Exact complete saved sources, cursor, model specification and metrics required')
        for part, module in [('network', meta.network), ('heads', meta.heads)]:
            expected = module.state_dict()
            if (saved[part].keys() != expected.keys() or any(t.shape != expected[n].shape
                    or t.dtype != expected[n].dtype or not bool(torch.isfinite(t).all()) for n, t in saved[part].items())):
                raise ValueError('Every actual saved tensor must have its finite original declared shape/dtype')
        if (saved['network']['embedding.weight'].data_ptr() != saved['network']['lm_head.weight'].data_ptr()
                or any(not torch.equal(saved['network'][n], initial['network'][n]) for n in fixed)
                or any(not torch.equal(t, initial['heads'][n]) for n, t in saved['heads'].items())):
            raise ValueError('One actually tied vocabulary and every fixed vision/critic/head value required')
        _validate_adam(state['optimizer'], [saved['network'][n] for n in names], step)
        _validate_rng(state['rng'], 'cpu')
        for record in state['metrics']:
            denominator = math.fsum(plan['coefficients'][r['family']]*r['response_EOS_targets'] for r in information)
            if (record['source_positions'] != positions or record['original_source_records'] != information
                    or record['weighted_target_normalizer'] != denominator or record['reference_queried'] is not True):
                raise ValueError('Every original source and weighted complete target denominator must reconstruct')
            for name, item in record['streams'].items():
                info = next(r for r in information if r['family'] == name)
                if (item['examples'] != 1 or item['complete_context_tokens'] != info['complete_context_tokens']
                        or item['response_EOS_targets'] != info['response_EOS_targets']
                        or item['coefficient'] != plan['coefficients'][name] or item['reference_route'] != REFERENCE_ROUTES[name]
                        or any(not math.isfinite(item[k]) for k in ('CE_sum','reference_KL_sum'))
                        or item['token_mean_CE'] != item['CE_sum']/item['response_EOS_targets']
                        or item['token_mean_reference_KL'] != item['reference_KL_sum']/item['response_EOS_targets']):
                    raise ValueError('Every family complete mask, reference route and CE/KL sum required')
            objective = math.fsum(r['coefficient']*(r['CE_sum']+plan['reference_KL_alpha']*r['reference_KL_sum'])
                                  for r in record['streams'].values())/denominator
            if record['summed_objective'] != objective:
                raise ValueError('The actual complete weighted objective must reconstruct')
    if proof['own_reference_digest'] != settings['own_reference_state_sha256']:
        raise ValueError('Frozen prior own general/code reference required')
    guard.check(full=True)
    atomic_json(output, dict(reader_sha256=file_hash(__file__),complete_sha256=file_hash(REPORTS/'complete.json'),
        actual_counts=meta.counts(),actual_dense_language_values=931244544,actual_Adam_states=317,
        both_complete_model_Adam_CPU_RNG_source_cursors_and_objective_arithmetic_reconstructed=True,
        complete_original_selected_target_masks=source_masks,
        education_reader_rehashed_all_original_packed_shards_and_admission_evidence=True,
        all_saved_tensors_finite_exact_shapes_and_one_tied_vocabulary=True,
        all_fixed_vision_critic_auxiliary_heads_equal_original_initializer=True,
        original_probe_replayed_actual_next_update_and_cold_reload=True,
        reader_did_not_replay_neural_forward_or_optimizer=True,
        CPU_book_context256_not_production4096_capacity=True,
        full_source_training_and_held_quality_not_claimed=True,candidate_promoted=False,goal_achieved=False))
    print(output)


if __name__ == '__main__': main()
