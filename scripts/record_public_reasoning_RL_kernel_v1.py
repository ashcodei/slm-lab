"""Review tested reasoning warmup/reward gradients, not a production run."""
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from slm_lab.function_repair_GPU_contract_v1 import ADMISSION,QUEUE,handoff_admitted
from slm_lab.local_source_closure_v2 import python_sources

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/public-reasoning-RL-kernel-v1-review'
TESTS=ROOT/'reports/public-reasoning-group-gradients-v1-tests.log'
ADMITTED=ROOT/'reports/public-reasoning-curriculum-v1-readback-admission/complete.json'


def read(path):return json.loads(Path(path).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    pending.replace(path)


def live(owner):
    fields=(Path('/proc')/str(owner['pid'])/'stat').read_text().split(') ',1)[1].split()
    if fields[19]!=owner['ticks'] or fields[0] in {'Z','X'}:raise ValueError('Every original process birth must remain live')
    return dict(owner=owner,state=fields[0])


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists():raise ValueError('Fresh low-priority CPU kernel review required')
    if '9 passed' not in TESTS.read_text():raise ValueError('All actual data/reward/neural/recovery/epoch controls must pass')
    admitted=read(ADMITTED)
    if not admitted.get('complete') or admitted['actual_counts']['train']['examples']!=2568:raise ValueError('Exact independent full train-data admission required')
    audit=read(ROOT/'reports/completion-audit.json');previous=Path(audit['latest_full_scope_review']);review=copy.deepcopy(read(previous))
    gpu=read(ROOT/QUEUE);handoff=read(ROOT/ADMISSION)
    if not handoff_admitted(ROOT,gpu['owner']):raise ValueError('All ten complete original successor scopes must remain admitted')
    maps=[read(ROOT/'reports/public-pretraining-recovery-v1/plan.json')['source_files_sha256'],
        read(ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json')['source_files_sha256'],
        read(ROOT/'reports/public-pretraining-interim-CPU-v2/evaluation-plan.json')['source_files_sha256'],
        read(ROOT/'reports/public-edge-preference-comparison-CPU-v1-queue.json')['source_files_sha256'],gpu['source_files_sha256']]
    maps += [read(ROOT/item['queue'])['source_files_sha256'] for item in handoff['new_controllers']]
    for name in ['reports/completion-audit.json','README.md','RESEARCH.md']:
        if any(str(ROOT/name) in mapping for mapping in maps):raise ValueError('Never edit a live source-bound index/document')
    files=dict(admitted['source_files_sha256'])
    paths=[ADMITTED,TESTS,previous,ROOT/'data/public-reasoning-curriculum-v1/admission.json',
        ROOT/'reports/public-reasoning-group-gradients-v1-initial-tests.log',
        ROOT/'reports/public-reasoning-group-gradients-v1-eight-controls-tests.log',
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_public_reasoning_group_gradients_v1.py'])]
    files.update({str(path):sha(path) for path in paths})
    if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('All admitted source and new tested kernel bytes must remain exact')
    raw=(ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes();prefix=raw[:raw.rfind(b'\n')+1]
    rows=list(map(json.loads,prefix.splitlines()))
    if [row['step'] for row in rows]!=list(range(1,len(rows)+1)):raise ValueError('Contiguous complete original neural update prefix required')
    current=dict(owner=dict(pid=1241,ticks='49351203'),complete_actual_updates=len(rows),planned_updates=4447,
        response_EOS_targets=sum(int(row['response_EOS_targets']) for row in rows),complete_prefix_sha256=hashlib.sha256(prefix).hexdigest())
    actual=[live(owner) for owner in [current['owner'],dict(pid=18545,ticks='50128787'),dict(pid=25240,ticks='51635952'),
        gpu['owner'],read(ROOT/'reports/public-edge-preference-comparison-CPU-v1-queue.json')['owner']]]
    if any(row['state'] in {'T','t'} for row in actual):raise ValueError('No active original neural/research worker may be stopped')
    at=datetime.now(timezone.utc).isoformat();OUT.mkdir()
    proof=dict(complete=True,at=at,source_files_sha256=files,
        previous_goal_turn_classification='progress: full3072source/protected31349input/native2857mask readback and actual2568train-only admission',
        current_goal_turn_classification='progress: actual train-only native reward adapter and CE/group-policy gradient kernels, nine meaningful source/neural/recovery controls',
        actual_focused_tests_passed=9,actual_admitted_train_rows_checked_with_true_EOS_positive_rewards=2568,
        actual_whole_source_schedule_control=dict(warmup_epochs=3,warmup_updates=483,source_presentations=7704,
            complete_context_tokens=735570,response_EOS_targets=57240,whole_single_source_epoch_updates=2568,
            single_source_epoch_positive_targets=19080,maximum_group4sampled_actions_in_whole_epoch=968448),
        neural_control_scope='small actual own-neural configurations; mixed/all-zero sampling uses explicitly hand-initialized two-action fixture weights',
        actual_FP32_and_CPU_BF16_cached_warmup_next_update_Adam_RNG_and_cold_controls_passed=True,
        actual_fresh_full_vocabulary_sampling_and_reward_dependent_gradients_checked=True,
        actual_identical_zero_reward_group_has_zero_policy_advantages_and_retains_complete_CE=True,
        actual_partial_source_failure_clears_uncommitted_gradients=True,
        production_model_has_consumed_new_curriculum=False,new_production_model_updates=0,
        production_driver_and_full_size_capacity_recovery_throughput_and_native_quality_still_pending=True,
        new_production_GPU_or_CPU_training_queue_created=False,original_main=current,actual_live_original_processes=actual,
        original_CPU_comparison=read(ROOT/'reports/full-function-repair-comparison-CPU-v1/progress.json'),
        original_interim_assessment=read(ROOT/'reports/public-pretraining-interim-CPU-v2/progress.json'),
        actual_GPU_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True),
        all_original17family_human_RL_and_nine_endpoint_scopes_preserved=True,candidate_promoted=False,goal_achieved=False)
    atomic(OUT/'complete.json',proof)
    lines=['# Tested public reasoning warmup and verifier-gradient kernels','',
        'The previous turn admitted 2,568 whole public procedural train examples and kept 289 source-grouped representatives reserved. '
        'A new train-only adapter now reconstructs every actual source, whole prompt/answer mask and independent answer before exposing it for optimization.', '',
        'The new warmup kernel supervises every original response token and true EOS. The whole three-epoch schedule contains **483 updates**, '
        '**7,704 source presentations**, **735,570 complete context tokens**, and **57,240 response/EOS targets**. '
        'A subsequent full one-source epoch has 2,568 updates and 19,080 positive targets per arm. These are checked schedule budgets, not completed production updates.', '',
        'The verifier-gradient kernel samples a fresh group from the unchanged own model at temperature1 over the entire vocabulary. '
        'It retains every actual token and probability, independently rescores cached behavior against the full forward, and compares the policy to a frozen snapshot of its own starting actor. '
        'The current implementation requires FP32/TF32-disabled sampling, with an explicit maximum probability discrepancy of0.001. '
        'This is an ordinary fresh single-update group-relative REINFORCE variant with token-count normalization and full-vocabulary own-reference KL. '
        'It is not a claim of global novelty or an exact implementation of published GRPO. '
        'Background: [DeepSeekMath](https://arxiv.org/abs/2402.03300) introduces GRPO; '
        '[Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783) examines base-model dependence and normalization biases.', '',
        'Both anchor_CE and anchor_RL observe their actual fresh samples and retain complete public-answer CE regardless of reward. '
        'Only anchor_RL applies the reward-dependent policy term. Equal-reward groups have zero relative policy advantages, including all-failed groups; '
        'the supervised signal remains. Every generated response is delivered unchanged to the strict non-executing verifier. '
        'The protocol requires canonical text, actual terminal EOS and no other emitted special tokens. A horizon stop is retained as such and never repaired.', '',
        '**Nine actual controls pass:** all 2,568 original train sources receive a correct positive reward; invalid special tokens, arbitrary Python text, '
        'unfinished responses and impossible trajectories are rejected without output execution or repair; FP32 and cached CPU BF16 warmup pass exact selected-Adam/RNG '
        'next-update and cold-output checks; mixed and all-zero actual sampled groups exercise different and zero policy gradients respectively; '
        'invalid settings do not sample or mutate parameters; a later source failure clears all uncommitted gradients; and complete epoch budgets reconstruct.', '',
        'The neural controls use small explicit own-model configurations. The mixed/all-zero fixtures deliberately initialize two vocabulary actions so these '
        'branches can actually be exercised. Their token and reward outcomes are software-mechanics evidence, **not production-model accuracy**. '
        'An initial fixed fixture seed yielded only zero-reward samples; that failed test log is preserved and a separate fixed test seed exercises mixed outcomes. '
        'Production sampling contains no resampling-until-success branch. No small-model result establishes full-size capacity or throughput.', '',
        '**This is a tested gradient/data kernel, not a launched production training stage.** '
        'The next integration work must bind complete source schedules and optimizer/checkpoint cursors, complete actual full-size warmup and native64/256-horizon recovery/capacity checks, '
        'and retain matched native reserved/source/code/broad evaluation before any adoption. '
        'The full original four-GPU pipeline continues and no competing GPU lease or new large CPU neural job is started.', '',
        f'The original main remains live at **{len(rows):,}/4,447** updates with **{current["response_EOS_targets"]:,}** actual targets. '
        'All original 17 families, human feedback, execution RL, machine preferences and nine-endpoint assessments remain unchanged. '
        '**No production model consumed the new curriculum; new production updates are0. Latest complete local functional correctness remains0/48. '
        'No new capability gain or frontier parity is established; all ten final requirements remain incomplete.**','']
    (OUT/'metrics.md').write_text('\n'.join(lines))
    review['requirements']['serious_full_model_training']['evidence']+=(
        ' A new train-only public reasoning adapter and complete CE/fresh verifier-group gradient kernel pass9actual controls. '
        'All2568train sources have native true-EOS reward reconstruction; complete3epoch budgets are483updates/7704presentations/57240targets. '
        'Small own-neural FP32/BF16 warmup passes exact next-update selectedAdam/RNG/cold controls. '
        'Explicit small two-action fixtures test actual fresh full-vocabulary mixed/all-zero reward groups, policy-gradient contribution and retained whole-answer CE. '
        'Partial failures clear gradients; emitted specials and unfinished answers are never repaired. '
        'These are mechanics only:0new production updates, no production queue, and actual full-size controller/capacity/recovery/throughput/native-quality work remains.')
    if len(review['requirements'])!=10 or any(v['status']!='incomplete_for_requested_final_model' for v in review['requirements'].values()):
        raise ValueError('Kernel controls prove none of the requested final model capabilities')
    review.update(at=at,previous_full_scope_review=str(previous),previous_goal_turn_classification=proof['previous_goal_turn_classification'],
        current_goal_turn_classification=proof['current_goal_turn_classification'],latest_actual_reasoning_RL_kernel=proof,
        source_files_sha256={**files,str(OUT/'complete.json'):sha(OUT/'complete.json')},goal_achieved=False)
    atomic(OUT/'review.json',review)
    if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('All original and new frozen sources must remain exact before index update')
    audit.update(at=at,requirements=review['requirements'],latest_full_scope_review=str(OUT/'review.json'),latest_actual_main_runtime=current,
        latest_actual_reasoning_RL_kernel=str(OUT/'complete.json'),goal_achieved=False)
    atomic(ROOT/'reports/completion-audit.json',audit)
    paragraph=(f'Reasoning warmup/verifier kernels tested, **{at}**: **9actual source/neural/recovery controls pass**. '
        'The train-only adapter checks all2,568whole sources. FP32/BF16 warmup passes exact next-update Adam/RNG and cold controls; '
        'small explicit own-neural fixtures test fresh full-vocabulary mixed/all-zero reward groups and retained complete CE. '
        '**These are mechanics tests, not full-size capacity or model accuracy. No new production queue or model updates exist.** '
        f'The original main continues at **{len(rows)}/4,447**, with all original source/human/RL/nine-endpoint scopes. '
        '**Latest complete functional correctness remains0/48; all ten final requirements remain incomplete.** '
        '[Actual controls and pending integration](reports/public-reasoning-RL-kernel-v1-review/metrics.md), '
        '[full review](reports/public-reasoning-RL-kernel-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md','RESEARCH.md']:
        path=ROOT/name;first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),actual_tests_passed=9,new_production_updates=0,main_actual_updates=len(rows),goal_achieved=False)),flush=True)


if __name__=='__main__':main()
