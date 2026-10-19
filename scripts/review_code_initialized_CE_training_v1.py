"""Read the completed CE arm while its separate KL comparison continues.

This checks every saved train draw, selected tensor/Adam state and complete
held teacher target. It does not replay training or claim generated coding
quality, and never reads a still-being-written KL checkpoint.
"""
from collections import Counter
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/code-initialized-alignment-v1'
OUTPUT=ROOT/'checkpoints/code-initialized-alignment-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('The independent completed-arm reader must hide every GPU')
    destination=REPORTS/'independent-complete-CE-training-and-held-targets-v1.json'
    if destination.exists():raise ValueError('Preserve prior completed-arm evidence')
    import torch
    import train_code_initialized_alignment_v1 as driver
    from slm_lab.code_initialized_alignment_evidence_v1 import budget,positions,source_counts
    from slm_lab.code_initialized_alignment_inference_v1 import sources,options,FORMAT
    from slm_lab.code_residual_checkpoint_v1 import _validate_adam,_validate_rng
    from slm_lab.corpus import file_hash
    from slm_lab.cuda_rng_readback_v1 import review_cuda_rng_layout
    from slm_lab.diverse_images import atomic_json
    from slm_lab.model import ModelConfig
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    from review_whole_human_SFT_targets_v1 import review as review_held
    torch.set_num_threads(2)
    plan_path=REPORTS/'plan.json';sha=file_hash(plan_path);plan=json.loads(plan_path.read_text());arm='mixed_CE';setting=options(plan,arm)
    receipt=json.loads((REPORTS/(arm+'-training-complete.json')).read_text());checkpoint=OUTPUT/arm/'model.pt';resume=OUTPUT/arm/'resume.pt'
    if file_hash(checkpoint)!=receipt['checkpoint_sha256'] or file_hash(resume)!=receipt['resume_sha256']:
        raise ValueError('Exact final exported policy and complete optimizer snapshot required')
    initial=torch.load(OUTPUT/'initial/model.pt',map_location='cpu',weights_only=True,mmap=True)['aligned_model']['extensions']
    raw=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True);state=torch.load(resume,map_location='cpu',weights_only=True,mmap=True)
    core=state['core'];payload=raw['aligned_model'];parts=payload['extensions'];names=[n for n in initial if n.startswith(('language_linears.','language_norms.'))]
    if (raw['format']!=FORMAT or raw['arm']!=arm or raw['run_plan_sha256']!=sha or raw['actual_mixed_alignment_updates']!=3548
            or raw['training_options']!=setting or raw['actual_source_exposure_budget']!=plan['source_exposure_budget']
            or state['options']!=setting or core['steps']!=3548 or core['planned_steps']!=3548 or core['groups']!=['language']
            or core['optimized_names']!=names or len(core['metrics'])!=3548 or payload['sources']!=sources(plan,sha)
            or core['model']['sources']!=sources(plan,sha) or payload['actual_counts']!=plan['actual_counts']
            or parts.keys()!=initial.keys() or any(not torch.equal(t,core['model']['extensions'][n]) for n,t in parts.items())
            or any(not torch.equal(t,initial[n]) for n,t in parts.items() if n not in names)
            or any(t.shape!=initial[n].shape or t.dtype!=initial[n].dtype or not bool(torch.isfinite(t).all()) for n,t in parts.items())
            or tensor_digest(initial.items())!=plan['initial_parts_digest']):
        raise ValueError('Every completed actual policy/source/profile/unused tensor must reconcile')
    _validate_adam(core['optimizer'],[parts[n] for n in names],3548);_validate_rng({**core['rng'],'cuda':None},'cpu')
    cuda=core['rng']['cuda']
    cuda_layout=review_cuda_rng_layout(cuda)
    with (REPORTS/(arm+'-metrics.jsonl')).open() as f:journal=[json.loads(line) for line in f]
    if journal!=core['metrics']:raise ValueError('Every original flushed training record must equal saved history')
    streams=driver.streams(ModelConfig(**json.loads((ROOT/'configs/model.json').read_text())))
    expected=budget(streams,setting)
    if expected!=plan['source_exposure_budget'] or expected!=receipt['source_exposure_budget']:
        raise ValueError('Every source-complete training exposure budget required')
    draws={n:Counter() for n in streams};totals={n:Counter() for n in streams}
    for step,row in enumerate(journal):
        chosen=positions(streams,setting,step)
        if (row['step']!=step+1 or row['source_positions']!=chosen or row['learning_rate']!=setting['learning_rate']
                or row['reference_KL_alpha']!=0 or not math.isfinite(row['gradient_norm_before_clip']) or row['gradient_norm_before_clip']<0):
            raise ValueError('Every original mixed-source step and fixed objective profile required')
        objective=0.
        for name,selected in chosen.items():
            context,targets=source_counts(streams,name,selected);metric=row['streams'][name]
            if (metric['complete_context_tokens']!=context or metric['response_EOS_targets']!=targets
                    or metric['reference_route']!=('general' if name=='human' else 'code')
                    or not math.isfinite(metric['token_mean_CE']) or metric['token_mean_CE']<0 or metric['token_mean_reference_KL']!=0):
                raise ValueError('Every complete source context and response/EOS target denominator required')
            objective+=metric['token_mean_CE'];draws[name].update(selected);totals[name].update(context=context,targets=targets)
        if row['summed_objective']!=objective:raise ValueError('Every actual CE objective must reconstruct')
    cursor={}
    for i,(name,width) in enumerate([('human',4),('generation',1)]):
        count=3548*width;pop=len(streams[name]);cursor[name]=dict(source_population=pop,actual_draws=count,
            complete_passes=count//pop,next_pass_remainder=count%pop,seed=setting['seed']+i*1009)
        if len(draws[name])!=pop or totals[name]!=dict(context=expected[name]['complete_context_tokens'],targets=expected[name]['response_EOS_targets']):
            raise ValueError('All admitted original sources and token targets must appear')
    if core['sampler_state']!=cursor:raise ValueError('The exact final complete-source cursor required')
    valid=WholeHumanData(ROOT/'data/whole-human-alignment-v2','dialogue','valid','evaluation');held={}
    for candidate in ('initial',arm):
        path=REPORTS/candidate/'human-validation.json';held[candidate]=dict(metrics=review_held(valid,json.loads(path.read_text())),sha256=file_hash(path))
    initial_score,trained_score=held['initial']['metrics'],held[arm]['metrics'];reload_path=REPORTS/arm/'trained-shared-reload.json'
    atomic_json(destination,dict(observed_at=datetime.now(timezone.utc).isoformat(),reader_sha256=file_hash(Path(__file__)),run_plan_sha256=sha,
        completed_arm=arm,actual_updates=3548,actual_language_Adam_states=len(core['optimizer']['state']),
        actual_optimized_values=sum(parts[n].numel() for n in names),checkpoint_sha256=receipt['checkpoint_sha256'],resume_sha256=receipt['resume_sha256'],
        retained_CUDA_RNG_layout=cuda_layout,
        every_export_tensor_equal_complete_resume=True,every_unused_extension_equal_initial=True,
        every_original_training_draw_context_target_count_and_objective_reconciled=True,exact_final_source_cursors=True,
        source_exposures=expected,complete_held_teacher_source_target_readback=held,
        relative_held_token_NLL_change=trained_score['token_mean_nll']/initial_score['token_mean_nll']-1,
        teacher_argmax_accuracy_percentage_point_change=100*(trained_score['teacher_argmax_accuracy']-initial_score['teacher_argmax_accuracy']),
        original_worker_fixed_routes_and_cache_checks=receipt['all_fixed_bytes_versions_and_original_native_routes_exact'],
        trained_full_model_cold_reload_complete=reload_path.exists() and json.loads(reload_path.read_text())['full_trained_aligned_and_original_3routes_exact'],
        CPU_reader_did_not_replay_training_or_GPU_neural_held_scores=True,
        KL_arm_and_all_native_generated_coding_knowledge_math_image_comparisons_pending=True,candidate_promoted=False,goal_achieved=False))
    print(destination,flush=True)


if __name__=='__main__':main()
