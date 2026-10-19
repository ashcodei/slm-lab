"""Review tested preference mechanics and an actual resource-gated CPU queue."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.function_repair_GPU_contract_v1 import ADMISSION, QUEUE, handoff_admitted

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/public-edge-preference-readiness-v1-review'
STUDY=ROOT/'reports/public-edge-preference-comparison-CPU-v1-queue.json'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():
        raise ValueError('Fresh CUDA-hidden full-scope readiness review required')
    queue=read(STUDY)
    main_plan=read(ROOT/'reports/public-pretraining-recovery-v1/plan.json')
    cpu_plan=read(ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json')
    assessment=read(ROOT/'reports/public-pretraining-interim-CPU-v2/evaluation-plan.json')
    gpu_queue=read(ROOT/QUEUE)
    if (terminal(queue['owner']) or terminal(main_plan['owner']) or terminal(cpu_plan['owner'])
            or terminal(assessment['owner']) or not handoff_admitted(ROOT,gpu_queue['owner'])
            or queue['public_pretraining_owner']!=main_plan['owner']
            or queue['exact_two_thread_assessment_owner']!=assessment['owner']):
        raise ValueError('Every exact predecessor, actual new queue and original downstream owner must remain live')
    if (ROOT/'reports/public-edge-preference-comparison-CPU-v1').exists():
        raise ValueError('This readiness record must not claim that pending full-size capacity has run')
    tests=ROOT/'reports/public-edge-preference-stage-v1-tests.log'
    if '8 passed' not in tests.read_text():
        raise ValueError('Actual complete-source and small-neural control evidence required')
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if str(queue['owner']['pid']) in compute.splitlines():
        raise ValueError('The waiting CPU queue must not own a GPU context')
    audit=read(ROOT/'reports/completion-audit.json');previous=Path(audit['latest_full_scope_review'])
    files=dict(queue['source_files_sha256'])
    for path in [Path(__file__),STUDY,tests,previous]:
        files[str(path)]=file_hash(path)
    guard=FileBindings(files)
    raw=(ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes();prefix=raw[:raw.rfind(b'\n')+1]
    rows=list(map(json.loads,prefix.splitlines()))
    if [r['step'] for r in rows]!=list(range(1,len(rows)+1)):
        raise ValueError('Actual contiguous complete main prefix required')
    current=dict(owner=main_plan['owner'],complete_actual_updates=len(rows),planned_updates=4447,
                 response_EOS_targets=sum(int(r['response_EOS_targets']) for r in rows),
                 complete_prefix_sha256=hashlib.sha256(prefix).hexdigest())
    status=read(ROOT/'reports/public-edge-preference-comparison-CPU-v1-queue-progress.json')
    at=datetime.now(timezone.utc).isoformat();OUT.mkdir()
    proof=dict(complete=True,at=at,source_files_sha256=files,previous_goal_turn_classification='progress:77source-preserving hard preference pairs independently admitted',
               current_goal_turn_classification='progress: tested native first-divergence preference mechanism and actual gated full-size comparison queued',
               actual_new_queue_owner=queue['owner'],actual_new_queue_status=status,
               actual_small_neural_source_gradient_recovery_cold_tests_passed=8,
               actual_fixed_profiles=queue['fixed_profiles'],actual_future_updates_per_arm=60,
               actual_future_pair_presentations_per_arm=231,actual_future_positive_targets_per_arm=98106,
               actual_future_negative_response_EOS_observations_per_arm=98109,
               all_original15source48code_and_broad_assessment_budgets=queue['fixed_local_assessments'],
               original_main=current,original_CPU_comparison=read(ROOT/'reports/full-function-repair-comparison-CPU-v1/progress.json'),
               original_interim_assessment=read(ROOT/'reports/public-pretraining-interim-CPU-v2/progress.json'),
               actual_GPU_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used',
                                                       '--format=csv,noheader,nounits'],text=True),
               actual_full_size_capacity_passed=False,new_full_size_neural_optimizer_updates=0,
               no_GPU_lease_or_neural_worker_signalled=True,original_canonical_human_RL_export_and_final_benchmark_scope_preserved=True,
               new_generated_quality_gain_established=False,candidate_promoted=False,goal_achieved=False)
    atomic_json(OUT/'complete.json',proof)
    (OUT/'metrics.md').write_text(
        '# First-divergence preference comparison readiness\n\n'
        'The new comparison uses all 77independently admitted single-token code preference pairs from 23original public train functions. '
        'It compares whole-function chosen CE, CE plus whole-response DPO, and CE plus a preference at the first differing BPE token. '
        'That decision is scored using exactly the same original task and shared code prefix for both actions. '
        'Every arm observes both complete programs, their EOS targets, the shared prefix and the same frozen starting own actor/reference. '
        'Negative programs never become positive CE targets. No external model or human ranking is used.\n\n'
        '**Eight tests pass.** Real admitted data cover all 77pairs and all three complete epochs. Small own neural models exercise '
        'all three arms in FP32 and CPU BF16, finite gradients on only selected actor parameters, unchanged base/heads/critic, '
        'exact next-update weights/Adam/private-RNG recovery, pre-mutation rejection of corrupt masks/RNG/options, and cold policy logits. '
        'The analytic branch test verifies that its preference credits only the two critical next-token actions.\n\n'
        f'The actual queue owner is PID {queue["owner"]["pid"]}, birth {queue["owner"]["ticks"]}. It is CUDA-hidden and owns no GPU context. '
        'It waits for the exact live two-thread interim assessment to finish and for every 4,447 public-pretraining update, native controls '
        'and original worker restoration to complete. It then claims only the freed two CPU threads at nice 19. '
        'A 32 GiB working allowance, 64 GiB other-job headroom and the original 26 GiB disk reserve plus 14 GiB canonical writer allowance are retained.\n\n'
        'Actual full-size capacity remains pending. All three methods must pass full-size gradients, all 561 actor Adam leaves, '
        'next-update recovery and cold import before any of the 60 production updates per arm. Each arm retains all 231 pair presentations, '
        '98,106 chosen response/EOS targets and 98,109 negative response/EOS observations. The four actual baseline/final actor graphs '
        'retain all 15 source-validation functions at 1,024 tokens, 48 local code tasks at 512 tokens and the same 56 knowledge/16 math/20 OCR/'
        '15 object/32 photo development checks. All three arms finish training before those new evaluations.\n\n'
        'This is a local credit-assignment hypothesis and an actual queued experiment, not a globally novel algorithm or a measured capability gain. '
        'Source-validation independence from all historical exposure is not certified. The same actual actor graph defines the baseline and all arms; '
        'bit identity to a distinct base-forward graph is not claimed. The original 826 function/791 repair comparisons, all 17 pretraining families, '
        'full human preference/RL/export pipelines and every full final benchmark remain unchanged. '
        '**No new full-size optimizer update or accuracy gain is established. All ten final requirements remain incomplete.**\n')
    review=copy.deepcopy(read(previous))
    evidence=(' A new source-bound three-arm comparison now has 8 passing real-data/small-neural FP32/BF16 gradient, recovery and cold tests. '
              'All77finite-qualified machine pairs are fixed for 3 complete epochs/60 updates/98,106 positive targets per arm; '
              'ordinary CE, whole-response DPO and first-divergence preference retain identical original context observations. '
              'The actual CUDA-hidden controller waits for the exact interim CPU owner and complete public predecessor, then reuses only 2 freed CPU threads. '
              'Actual full-size capacity/training and baseline/all 3 arms full 15 source/48 code/broad assessments are pending; '
              'no new full-size optimizer update, human RLHF, on-policy RL, broad accuracy gain or automatic adoption is claimed.')
    for key in ['serious_full_model_training','high_end_llm_capability','any_code_correctness_and_efficiency','human_feedback']:
        review['requirements'][key]['evidence']+=evidence
    if any(v['status']!='incomplete_for_requested_final_model' for v in review['requirements'].values()):
        raise ValueError('Mechanics and queued capacity do not complete a final requirement')
    review.update(at=at,previous_full_scope_review=str(previous),current_goal_turn_classification=proof['current_goal_turn_classification'],
                  latest_actual_edge_preference_readiness=proof,source_files_sha256={**files,str(OUT/'complete.json'):file_hash(OUT/'complete.json')},goal_achieved=False)
    atomic_json(OUT/'review.json',review);guard.check(full=True)
    maps=[main_plan['source_files_sha256'],cpu_plan['source_files_sha256'],assessment['source_files_sha256'],
          queue['source_files_sha256'],gpu_queue['source_files_sha256']]
    maps += [read(ROOT/item['queue'])['source_files_sha256'] for item in read(ROOT/ADMISSION)['new_controllers']]
    for name in ['reports/completion-audit.json','README.md','RESEARCH.md']:
        if any(str(ROOT/name) in mapping for mapping in maps):
            raise ValueError('Never change source-bound live documents')
    audit.update(at=at,latest_full_scope_review=str(OUT/'review.json'),requirements=review['requirements'],
                 latest_actual_main_runtime=current,latest_edge_preference_readiness=str(OUT/'complete.json'),goal_achieved=False)
    atomic_json(ROOT/'reports/completion-audit.json',audit)
    paragraph=(f'First-divergence coding preference readiness, **{at}**: 8 tests with real data and small neural models tests pass for ordinary CE, '
               'whole-response DPO and a preference at the first differing code token. A real CUDA-hidden controller is queued behind '
               'the exact two-thread interim assessment and complete public pretraining. Every arm retains all 77 pairs, 3 epochs, 60 updates '
               'and 98,106 positive targets, then full 15 source/48 code/broad development checks. It acquires no GPU lease and preserves '
               f'all canonical human/RL/final-benchmark stages. The original main is at **{len(rows)}/4,447**. '
               '**Actual full-size capacity/training are pending; no accuracy gain is established. All ten final requirements remain incomplete.** '
               '[Readiness evidence](reports/public-edge-preference-readiness-v1-review/metrics.md), '
               '[full review](reports/public-edge-preference-readiness-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md','RESEARCH.md']:
        path=ROOT/name;first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),actual_queue_owner=queue['owner'],new_full_size_optimizer_updates=0,goal_achieved=False)),flush=True)


if __name__=='__main__':main()
