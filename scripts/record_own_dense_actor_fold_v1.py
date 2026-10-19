"""Review measured adapter folding without changing the active model lineage."""
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

from record_public_code_instruction_screen_v1 import sha,read,atomic,live
from slm_lab.function_repair_GPU_contract_v1 import ADMISSION,QUEUE,handoff_admitted

ROOT=Path(__file__).resolve().parents[1]
MEASURE=ROOT/'reports/own-dense-actor-fold-CPU-v1'
OUT=ROOT/'reports/own-dense-actor-fold-v1-review'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists():raise ValueError('Fresh low-priority CPU-only review required')
    measured=read(MEASURE/'complete.json');audit=read(ROOT/'reports/completion-audit.json')
    previous=Path(audit['latest_full_scope_review']);review=copy.deepcopy(read(previous))
    if (not measured.get('complete') or measured['actual_mechanics_tests_passed']!=7
            or measured['actual_forward_2D_linear_calls']!={'actor':44,'folded':16}
            or measured['actual_cached_decode_2D_linear_calls']!={'actor':44,'folded':16}
            or measured['production_models_loaded_or_modified'] or measured['production_training_or_reset_job_started']):
        raise ValueError('Complete actual small learned parity and operator-call evidence required')
    gpu=read(ROOT/QUEUE);handoff=read(ROOT/ADMISSION)
    if not handoff_admitted(ROOT,gpu['owner']):raise ValueError('All ten exact original successors must remain admitted')
    plan_paths=[ROOT/'reports/public-pretraining-recovery-v1/plan.json',
        ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json',
        ROOT/'reports/public-pretraining-interim-CPU-v2/evaluation-plan.json',
        ROOT/'reports/public-edge-preference-comparison-CPU-v1-queue.json',
        ROOT/'reports/public-reasoning-warmup-CPU-v1-queue.json',ROOT/QUEUE,
        *[ROOT/item['queue'] for item in handoff['new_controllers']]]
    plans=[read(p) for p in plan_paths];maps=[p['source_files_sha256'] for p in plans]
    for name in ['reports/completion-audit.json','README.md','RESEARCH.md']:
        if any(str(ROOT/name) in m for m in maps):raise ValueError('Never edit an active source-bound index or document')
    files=dict(measured['source_files_sha256'])
    for mapping in maps:
        for path,digest in files.items():
            if path in mapping and mapping[path]!=digest:raise ValueError('The tested shared runtime bytes must match active source bindings')
    for p in [Path(__file__),previous,MEASURE/'complete.json',MEASURE/'plan.json',*plan_paths]:files[str(p)]=sha(p)
    if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every tested and original byte must remain exact')
    original=[live(p['owner']) for p in plans[:6]]
    if any(r['state'] in {'T','t'} for r in original[:3]):raise ValueError('No active neural worker may be stopped')
    raw=(ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes();prefix=raw[:raw.rfind(b'\n')+1]
    rows=[json.loads(x) for x in prefix.split(b'\n') if x]
    if [r['step'] for r in rows]!=list(range(1,len(rows)+1)):raise ValueError('All actual main updates must form a complete contiguous prefix')
    current=dict(owner=plans[0]['owner'],complete_actual_updates=len(rows),planned_updates=4447,
        response_EOS_targets=sum(int(r['response_EOS_targets']) for r in rows))
    at=datetime.now(timezone.utc).isoformat();OUT.mkdir()
    fp32=max(r['FP32_max_abs']['logits'] for r in measured['actual_text_and_image_parity'])
    bf16=max(r['BF16_max_abs']['logits'] for r in measured['actual_text_and_image_parity'])
    proof=dict(complete=True,at=at,source_files_sha256=files,
        previous_goal_turn_classification='progress: whole public synthetic source screen/readback and exact legal counterexample; all new data unadmitted',
        current_goal_turn_classification='progress: nonmutating own-policy folding and recycling with7actual small learned controls and observed44to16linear calls; no production model change',
        actual_small_learned_fixture_proof=str(MEASURE/'complete.json'),actual_mechanics_tests_passed=7,
        actual_small_forward_and_cached_decode_linear_calls=dict(actor=44,folded=16),
        actual_small_FP32_max_abs_logit_error=fp32,actual_small_BF16_max_abs_logit_error=bf16,
        all_auxiliary_head_values_and_vision_word_paths_checked=True,
        source_parameters_flags_pointers_grads_and_rng_preserved=True,policy_critic_preserved=True,
        fresh_optimizer_and_reference_binding_required_for_recycling=True,
        shape_derived_full_size_folded_parameters=measured['shape_derived_full_size_folded_parameter_count'],
        shape_derived_full_size_removed_values=measured['shape_derived_full_size_removed_parameter_count'],
        shape_derived_full_size_removed_linear_calls_per_forward_or_decode=measured['shape_derived_full_size_removed_linear_calls_per_forward_or_decode'],
        full_size_folding_capacity_generation_quality_and_latency_unverified=True,
        low_rank_post_training_bottleneck_is_hypothesis_not_explanation_proven_by_these_tests=True,
        research_prior_art=['https://arxiv.org/abs/2307.05695','https://arxiv.org/abs/2509.12960'],
        established_merging_not_claimed_globally_novel_or_guaranteed_better=True,
        production_models_loaded_or_modified=False,production_optimizer_updates_from_this_work=0,
        new_production_training_or_reset_controller_started=False,all_original17family_human_RL_and_nine_endpoint_scopes_preserved=True,
        actual_original_live_owners=original,actual_original_main=current,
        actual_GPU_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True),
        latest_complete_local_functional_correctness='0/48',new_capability_gain_established=False,
        candidate_promoted=False,goal_achieved=False)
    atomic(OUT/'complete.json',proof)
    text=(f'Own dense policy folding mechanics, {at}. A new CPU FP32 utility folds learned low-rank projection updates, policy norms and the policy critic '
        'into an independent copy of our own dense model. It preserves every vision/word/reward/pointing/router/detector weight, the tied vocabulary and the original source graph. '
        'An optional recycle utility creates fresh zero-output factors and preserves the learned critic; it explicitly requires a new optimizer and newly bound reference before future training.\n\n'
        'Seven actual small-own-network checks pass: learned text/image/masked/cached paths, critic, all head weights, word/detection outputs, cold existing-format import, '
        'ambient autocast protection, fresh-optimizer updates and invalid/active-source/resource rejection. '
        f'The independent small learned control measures 44 versus 16 two-dimensional linear calls in both full forward and cached decode. '
        f'Maximum FP32 logit error is {fp32:.10g}; BF16 error is {bf16:.10g}. '
        'The fixture has zero argmax mismatches at its observed positions; this is not full-sequence generation parity or a code score.\n\n'
        'From the existing full-size inventory, folding would remove 15,306,752 adapter/critic values and leave 984,399,690 parameters. '
        'Its35-layer shape would eliminate490factor linear calls per forward/decode. These are shape-derived projections, not an actual full-size fold or latency measurement. '
        'Full-size memory, text4096/photo448/word precision, cold recovery, generated-code/general/multimodal quality and serving speed must still pass before deployment or cycle training.\n\n'
        'Weight merging and restart training are established prior work. '
        '[ReLoRA](https://arxiv.org/abs/2307.05695) motivates repeated low-rank updates; '
        '[a later11M–66M SLM study](https://arxiv.org/abs/2509.12960) reports worse outcomes than full-rank training. '
        'Neither paper proves an improvement for this984Mmodel. We do not install a reset schedule or change any active training/evaluation job.\n\n'
        f'The original main is live at {len(rows):,}/4,447 updates and {current["response_EOS_targets"]:,} targets; '
        'all original source/human/RL/nine-endpoint scopes remain. Latest complete local coding correctness remains0/48 on an older checkpoint; '
        'the newer fixed interim evaluation is incomplete. No capability gain or frontier parity is established. All ten final requirements remain incomplete.\n')
    (OUT/'metrics.md').write_text(text)
    if len(review['requirements'])!=10 or any(v['status']!='incomplete_for_requested_final_model' for v in review['requirements'].values()):
        raise ValueError('Small folding mechanics establish none of the requested final capabilities')
    review['current_actual_runtime']['main']=current
    review.update(at=at,previous_full_scope_review=str(previous),previous_goal_turn_classification=proof['previous_goal_turn_classification'],
        current_goal_turn_classification=proof['current_goal_turn_classification'],latest_actual_own_dense_actor_fold=proof,
        source_files_sha256={**files,str(OUT/'complete.json'):sha(OUT/'complete.json')},goal_achieved=False)
    atomic(OUT/'review.json',review)
    if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every bound artifact must remain exact before index update')
    audit.update(at=at,requirements=review['requirements'],latest_full_scope_review=str(OUT/'review.json'),
        latest_actual_main_runtime=current,latest_actual_own_dense_actor_fold=str(OUT/'complete.json'),goal_achieved=False)
    atomic(ROOT/'reports/completion-audit.json',audit)
    paragraph=(f'Own-policy folding mechanics verified, **{at}**: **7 small learned controls pass**. '
        'Actual small full-forward and cached-decode linear calls fall **44→16**; copied vision/word/auxiliary weights and source state remain intact. '
        '**This is structural/mechanical evidence, not a full-size speed or capability gain**; FP32/BF16 rounding is measured and production adoption/reset training remains pending. '
        f'The original main continues at **{len(rows)}/4,447**. **Latest complete coding correctness remains0/48 on an older checkpoint; all ten final requirements remain incomplete.** '
        '[Measured scope and limits](reports/own-dense-actor-fold-v1-review/metrics.md), [full review](reports/own-dense-actor-fold-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md','RESEARCH.md']:
        path=ROOT/name;first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(complete=True,review=str(OUT/'review.json'),actual_main_updates=len(rows),production_models_modified=False,goal_achieved=False)))


if __name__=='__main__':main()
