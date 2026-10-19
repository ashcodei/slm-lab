"""Read back every final SFT tensor, Adam moment and original source exposure.

This supplements held-target reconstruction while native capability tests are
still running. It neither repeats training nor declares the experiment done.
"""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch

from slm_lab.code_residual_checkpoint_v1 import _validate_adam, _validate_rng
from slm_lab.codenet_stage_v1 import EpochOrder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.whole_human_data_v2 import WholeHumanData
from slm_lab.whole_human_sft_evidence_v1 import budget, tensor_digest
from slm_lab.whole_human_sft_inference_v1 import FORMAT, sources

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/whole-human-SFT-v1'
OUTPUT = ROOT/'checkpoints/whole-human-SFT-v1'


def main():
    destination = REPORTS/'independent-complete-training-readback-v1.json'
    if destination.exists(): raise ValueError('Preserve prior whole training readback')
    torch.set_num_threads(2)
    plan_path = REPORTS/'plan.json'; plan_sha = file_hash(plan_path); plan = json.loads(plan_path.read_text())
    data = WholeHumanData(ROOT/'data/whole-human-alignment-v2', 'dialogue', 'train', 'optimization')
    initial = torch.load(OUTPUT/'initial/model.pt', map_location='cpu', weights_only=True, mmap=True)
    initial_parts = initial['aligned_model']['extensions']
    if tensor_digest(initial_parts.items()) != plan['initial_parts_digest']:
        raise ValueError('Exact fresh seeded initial model extensions required')
    result = {}
    for arm, spec in plan['arms'].items():
        receipt = json.loads((REPORTS/(arm+'-training-complete.json')).read_text())
        exported_path = OUTPUT/arm/'model.pt'; resume_path = OUTPUT/arm/'resume.pt'
        if file_hash(exported_path) != receipt['checkpoint_sha256'] or file_hash(resume_path) != receipt['resume_sha256']:
            raise ValueError('Exact retained completed worker checkpoint bytes required')
        export = torch.load(exported_path, map_location='cpu', weights_only=True, mmap=True)
        resume = torch.load(resume_path, map_location='cpu', weights_only=True, mmap=True); core = resume['core']
        parts = export['aligned_model']['extensions']; options = {**plan['training_options'], 'learning_rate':spec['learning_rate']}
        names = [n for n in initial_parts if n.startswith(('language_linears.', 'language_norms.'))]
        steps = options['steps']; expected = budget(data, options)
        if (export['format'] != FORMAT or export['run_plan_sha256'] != plan_sha or export['arm'] != arm
                or export['aligned_model']['sources'] != sources(plan, plan_sha, arm)
                or core['model']['sources'] != sources(plan, plan_sha, arm)
                or export['actual_human_SFT_updates'] != steps or core['steps'] != steps or core['planned_steps'] != steps
                or resume['actual_human_SFT_updates'] != steps or resume['options'] != options or resume['source_population'] != len(data)
                or export['training_options'] != options or export['actual_source_exposure_budget'] != expected
                or expected != plan['source_exposure_budget'] or core['optimized_names'] != names
                or len(core['metrics']) != steps or set(parts) != set(initial_parts)
                or any(not bool(torch.isfinite(t).all()) or not torch.equal(t, core['model']['extensions'][n]) for n,t in parts.items())
                or any(not torch.equal(t, initial_parts[n]) for n,t in parts.items() if n not in names)):
            raise ValueError('Every completed source-bound trained tensor and unchanged unused extension must reconcile')
        _validate_adam(core['optimizer'], [parts[n] for n in names], steps)
        _validate_rng({**core['rng'], 'cuda':None}, 'cpu')
        cuda_rng = core['rng']['cuda']
        if not isinstance(cuda_rng, torch.Tensor) or cuda_rng.dtype != torch.uint8 or cuda_rng.ndim != 1 or not len(cuda_rng):
            raise ValueError('Actual retained CUDA RNG bytes required; this CPU reader does not instantiate a CUDA generator')
        order = EpochOrder(len(data), options['seed'])
        for step, metric in enumerate(core['metrics']):
            positions = [p for group in order.rank_groups(step, 0, 1, options['microbatch'], options['accumulation']) for p in group]
            context = sum(data.rows[p]['native']['response']['length'] for p in positions)
            targets = sum(data.rows[p]['native']['response']['length']-data.rows[p]['native']['response']['target_start'] for p in positions)
            if (metric['source_positions'] != positions or metric['complete_context_tokens'] != context or metric['response_EOS_targets'] != targets
                    or metric['step'] != step+1 or metric['learning_rate'] != options['learning_rate']
                    or any(not math.isfinite(metric[k]) or metric[k] < 0 for k in ('token_mean_nll', 'gradient_norm_before_clip'))):
                raise ValueError('Every complete original source draw, target normalizer and loss must reconstruct')
        if (sum(m['response_EOS_targets'] for m in core['metrics']) != expected['response_EOS_targets']
                or sum(m['complete_context_tokens'] for m in core['metrics']) != expected['complete_context_tokens']):
            raise ValueError('All whole original context and response target exposure totals must reconcile')
        cold = json.loads((REPORTS/arm/'trained-shared-reload.json').read_text())
        if (not cold['full_trained_aligned_and_original_3routes_exact'] or not cold['one_backbone_loaded']
                or cold['actual_counts'] != plan['actual_counts'] or receipt['source_exposure_budget'] != expected
                or not receipt['all_fixed_bytes_versions_and_original_native_routes_exact'] or not receipt['pre_existing_original_code_cache_exact']):
            raise ValueError('Actual full-model trained reload and original route/cache preservation evidence required')
        result[arm] = dict(actual_updates=steps, every_original_source_draw_and_target_normalizer_reconstructed=True,
            actual_Adam_states=len(core['optimizer']['state']), changed_language_tensors=sum(not torch.equal(parts[n], initial_parts[n]) for n in names),
            all_unused_new_tensor_count=len(parts)-len(names), every_unused_new_tensor_equal_initial=True,
            all_trained_export_tensors_equal_complete_resume=True, source_exposure_budget=expected,
            checkpoint_sha256=receipt['checkpoint_sha256'], resume_sha256=receipt['resume_sha256'],
            actual_trained_full_model_reload= cold, recorded_fixed_original_digest=core['fixed_original_state_sha256'])
    atomic_json(destination, dict(observed_at=datetime.now(timezone.utc).isoformat(), run_plan_sha256=plan_sha,
        completed_original_human_SFT_training=result, numerical_training_gradients_not_independently_replayed=True,
        actual_GPU_next_update_recovery_proved_separately_in_run_capacity=True,
        matched_native_capability_comparison_not_completed_by_this_reader=True, candidate_promoted=False, goal_achieved=False))


if __name__ == '__main__':
    main()
