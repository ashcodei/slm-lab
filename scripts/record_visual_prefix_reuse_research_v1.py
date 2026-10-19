"""Record the failed optimization hypothesis and bounded split-backward result."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from record_public_code_instruction_screen_v1 import read,sha,atomic,live
from slm_lab.function_repair_GPU_contract_v1 import ADMISSION,QUEUE,handoff_admitted
from slm_lab.local_source_closure_v2 import python_sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/visual-prefix-reuse-research-v1-review'
DIAG = ROOT/'reports/visual-prefix-gradient-rounding-CPU-v1/complete.json'
TESTS = ROOT/'reports/own-actor-vision-prefix-split-backward-v1-tests.log'


def main():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19:
        raise ValueError('Fresh CUDA-hidden low-priority research recorder required')
    observation = read(DIAG)
    if (not observation.get('complete') or observation['training_admitted']
            or observation['actual_small_fixture_parameters'] != 64194
            or not observation['failed_strict_optimizer_parity_not_waived'] or '7 passed' not in TESTS.read_text()):
        raise ValueError('Actual original failed control, four mechanical comparison cases and seven split-backward tests required')
    measured = []
    for case in observation['actual_cases']:
        old,naive,split = case['arms']
        if ([a['vision_encoder_forward_calls'] for a in case['arms']] != [2,1,1]
                or any(a['loss'] != old['loss'] or a['caption_CE'] != old['caption_CE'] or a['matching_loss'] != old['matching_loss'] for a in case['arms'])
                or not all(a['all_fixed_weights_and_text_logits_unchanged'] for a in case['arms'])
                or split['gradient_difference_from_two_forward']['all_values'] != 36576
                or split['gradient_difference_from_two_forward']['opposite_nonzero_sign_values'] != 0):
            raise ValueError('Actual full small selected-gradient/weight observations and unchanged native losses required')
        if not case['BF16_autocast']:
            if (split['gradient_difference_from_two_forward']['maximum_absolute_difference'] != 0
                    or split['Adam_parameter_difference_from_two_forward']['maximum_absolute_difference'] != 0):
                raise ValueError('The measured FP32 split-backward control must reproduce its original update exactly')
        elif (naive['gradient_difference_from_two_forward']['opposite_nonzero_sign_values'] != 30
                or naive['Adam_parameter_difference_from_two_forward']['maximum_absolute_difference'] <= .001
                or not 0 < split['Adam_parameter_difference_from_two_forward']['maximum_absolute_difference'] < 1e-6):
            raise ValueError('Preserve the actual BF16 failure and remaining nonzero split-backward difference')
        measured.append(dict(BF16_autocast=case['BF16_autocast'],checkpointing=case['checkpointing'],
            actual_selected_values=36576,original_vision_forward_calls=2,reuse_vision_forward_calls=1,
            naive_gradient_difference=naive['gradient_difference_from_two_forward'],
            naive_Adam_difference=naive['Adam_parameter_difference_from_two_forward'],
            split_gradient_difference=split['gradient_difference_from_two_forward'],
            split_Adam_difference=split['Adam_parameter_difference_from_two_forward']))
    if len(measured) != 4:
        raise ValueError('Both precisions and both checkpoint modes must be retained')
    existing = read(ROOT/'reports/completion-audit.json');previous = Path(existing['latest_full_scope_review'])
    prior = read(previous);requirements = prior['requirements']
    if len(requirements) != 10 or any(v['status'] != 'incomplete_for_requested_final_model' for v in requirements.values()):
        raise ValueError('Small numerical research proves none of the final high-end capabilities')
    canonical = read(ROOT/QUEUE)
    if not handoff_admitted(ROOT,canonical['owner']):raise ValueError('The original complete ten-controller handoff must remain admitted')
    handoff = read(ROOT/ADMISSION)
    paths = [ROOT/'reports/public-pretraining-recovery-v1/plan.json',
        ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json',
        ROOT/'reports/public-pretraining-interim-CPU-v2/evaluation-plan.json',
        ROOT/'reports/public-edge-preference-comparison-CPU-v1-queue.json',
        ROOT/'reports/public-reasoning-warmup-CPU-v1-queue.json',ROOT/QUEUE,
        ROOT/'reports/full-public-code-shard-qualification-CPU-v1/plan.json',
        ROOT/'reports/full-multimodal-pixel-validation-CPU-v1/plan.json']
    plans = [read(p) for p in paths]
    maps = [p['source_files_sha256'] for p in plans]
    maps += [read(ROOT/item['queue'])['source_files_sha256'] for item in handoff['new_controllers']]
    for name in ['reports/completion-audit.json','README.md','RESEARCH.md']:
        if any(str(ROOT/name) in mapping for mapping in maps):raise ValueError('Never edit a source-bound document or status index')
    jobs = [live(p['owner']) for p in plans]
    if any(p['state'] in {'T','t'} for p in jobs[:3]):raise ValueError('Original neural workers must not be stopped')
    if os.sched_getscheduler(plans[-1]['owner']['pid']) != os.SCHED_IDLE:
        raise ValueError('The original exact pixel auditor must remain at its declared idle scheduling priority')
    files = dict(observation['source_files_sha256'])
    for path in [DIAG,TESTS,previous,ROOT/ADMISSION,*python_sources(ROOT,[Path(__file__),
            ROOT/'tests/test_own_actor_vision_prefix_split_backward_v1.py'])]:
        files[str(path)] = sha(path)
    if any(sha(p) != digest for p,digest in files.items()):raise ValueError('All failed and passing controls must stay exact')
    raw = (ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes()
    prefix = raw[:raw.rfind(b'\n')+1];rows = [json.loads(line) for line in prefix.split(b'\n') if line]
    if [r['step'] for r in rows] != list(range(1,len(rows)+1)):raise ValueError('An actual contiguous completed main prefix is required')
    current = dict(owner=plans[0]['owner'],complete_actual_updates=len(rows),planned_updates=4447,
        response_EOS_targets=sum(int(r['response_EOS_targets']) for r in rows),complete_prefix_sha256=hashlib.sha256(prefix).hexdigest())
    at = datetime.now(timezone.utc).isoformat();OUT.mkdir()
    proof = dict(complete=True,at=at,source_files_sha256=files,
        previous_goal_turn_classification='progress: exact idle-scheduled full physical source auditor launched and five pixel-protocol tests passed',
        current_goal_turn_classification='progress: actual numerical counterexample rejects naive prefix reuse; separate-backward variant passes seven small mechanical/recovery controls',
        actual_small_fixture_parameters=64194,actual_selected_fixture_parameters=36576,
        actual_failed_naive_parity_cases=4,actual_other_original_controls_passed=3,actual_split_backward_controls_passed=7,
        actual_four_case_numerical_observations=measured,actual_encoder_forward_calls_2_to_1=True,
        original_tolerances_not_loosened=True,FP32_measured_update_bitexact=True,BF16_update_not_bitexact=True,
        two_loss_branch_backwards_still_required=True,full_stage_wall_time_or_GPU_speed_not_measured=True,
        no_extra_learned_values_or_persistent_feature_cache_or_model_monkeypatch=True,
        actual_original_main=current,actual_original_live_jobs=jobs,
        actual_pixel_audit_progress=read(ROOT/'reports/full-multimodal-pixel-validation-CPU-v1/progress.json'),
        actual_code_data_progress=read(ROOT/'reports/full-public-code-shard-qualification-CPU-v1/progress.json'),
        actual_GPU_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True),
        full_size_CUDA_distributed_capacity_recovery_and_matched_visual_quality_pending=True,
        actual_new_production_model_or_training_recipe_or_neural_queue_changes=0,
        all_original17family_human_RL_and_nine_endpoint_scopes_preserved=True,
        latest_completed_local_functional_correctness='0/48 on older checkpoint',newer_matched_endpoint_scores_pending=True,
        candidate_promoted=False,training_admitted=False,new_production_optimizer_updates=0,goal_achieved=False)
    atomic(OUT/'complete.json',proof)
    (OUT/'metrics.md').write_text(
        f'Visual-prefix reuse research, {at}. All observations use an actual 64,194-parameter own CPU fixture '
        'with 36,576 selected vision/pointing values, three complete image-caption examples and no external weights. '
        'This is a mechanical optimization experiment, not a full-size or model-quality result.\n\n'
        'The established objective calls the image encoder twice for caption CE and image/text matching. '
        'Naive shared-prefix/joint-backward reuse calls it once and produces identical losses, yet fails four '
        'strict gradient/Adam comparison cases. In BF16 it flips 30 nonzero gradient signs and changes one Adam '
        'weight by up to 0.001998478. The failed source/tests/log are preserved; that variant is not promoted.\n\n'
        'A separate-backward variant retains one forward and computes each loss branch backward separately. '
        'Both FP32 cases reproduce all selected gradients and the first Adam update bit exactly. In both BF16 '
        'cases the gradient relative L2 difference remains 0.00173645 (0.173645%), with zero observed opposite '
        'nonzero gradient signs; the largest Adam weight difference is 6.6869e-7. **BF16 is not bit exact.** '
        'All four forward losses match. Checkpointing on/off preserves these fixture observations.\n\n'
        '**Seven new controls pass** using the original comparison tolerances: four numerical/gradient/optimizer '
        'cases, two exact next-update Adam recovery cases with changed pixels, and stale-gradient/missing-coefficient '
        'rejection. Fixed language/actor/critic/other-head weights and text logits remain unchanged. No extra '
        'parameters, cross-update feature cache or temporary source-model patch is introduced.\n\n'
        'Two backwards remain; one fewer encoder forward does not establish a 2x training speedup. Full-size '
        'CUDA/distributed/capacity/recovery checks and full matched quality comparison remain required before '
        'any production adoption. The current model and all queued neural recipes are untouched. The whole '
        '195,121-path physical image audit continues independently at idle priority.\n\n'
        f'The original main continues at {len(rows):,}/4,447 committed updates and '
        f'{current["response_EOS_targets"]:,} response/EOS targets. Original source/human/RL/nine-endpoint scopes '
        'remain. Latest completed local code correctness is 0/48 on an older checkpoint; newer matched scores '
        'remain pending. No new accuracy gain or frontier parity is established. All ten final requirements '
        'remain incomplete.\n')
    review = dict(at=at,requirements=requirements,previous_full_scope_review=str(previous),
        current_goal_turn_classification=proof['current_goal_turn_classification'],
        current_actual_runtime=dict(main=current,original_live_jobs=jobs),
        latest_actual_visual_prefix_reuse_research=str(OUT/'complete.json'),
        all_original17family_human_RL_and_nine_endpoint_scopes_preserved=True,
        source_files_sha256={**files,str(OUT/'complete.json'):sha(OUT/'complete.json')},goal_achieved=False)
    atomic(OUT/'review.json',review)
    if any(sha(p) != digest for p,digest in files.items()):raise ValueError('Every actual failed/passing control must stay exact before updating indices')
    existing.update(at=at,requirements=requirements,latest_full_scope_review=str(OUT/'review.json'),
        latest_actual_main_runtime=current,latest_actual_visual_prefix_reuse_research=str(OUT/'complete.json'),goal_achieved=False)
    atomic(ROOT/'reports/completion-audit.json',existing)
    paragraph = (f'Visual-prefix reuse investigated, **{at}**: the naive variant fails4strict optimizer parity controls and is preserved unadopted. '
        '**Seven separate-backward controls pass** on a64,194parameter own CPU fixture: encoder forward calls2→1, exact FP32 updates, '
        'measured nonzero BF16 differences and exact self-recovery. **This is small mechanical evidence; no full-size/GPU speed or quality gain is established.** '
        f'The main continues at **{len(rows)}/4,447**; the full physical image audit remains active. '
        '**Latest completed local correctness remains0/48 on an older checkpoint; all ten final requirements remain incomplete.** '
        '[Actual failure, measurements and limits](reports/visual-prefix-reuse-research-v1-review/metrics.md), '
        '[full review](reports/visual-prefix-reuse-research-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md','RESEARCH.md']:
        path = ROOT/name;first,rest = path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(complete=True,split_backward_controls_passed=7,main_updates=len(rows),training_admitted=False,goal_achieved=False)))


if __name__ == '__main__':main()
