"""Record the actual three-arm research runtime without claiming quality gain."""
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
import own_training_gpu_lease_v2 as owner

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/full-function-repair-launch-v1-review'
RESEARCH=ROOT/'reports/full-function-repair-comparison-CPU-v1'


def read(path):return json.loads(Path(path).read_text())


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only full-scope review required')
    master_path=ROOT/'reports/completion-audit.json';master=read(master_path);previous=ROOT/master['latest_full_scope_review']
    requirements=deepcopy(read(previous)['requirements']);runtime=ROOT/'reports/public-pretraining-recovery-v1'
    plan=read(runtime/'plan.json');owner.REPORTS=runtime;owner.resources.REPORTS=runtime;owner.owned(plan)
    ranks=[r for r in read(runtime/'children.json') if r['subject'].startswith('train-rank-') and not terminal(r)]
    handoff=read(ROOT/'reports/instance-lineage-recovery-v2/complete.json')
    if len(ranks)!=4 or any(terminal(r['actual_owner']) for r in handoff['actual_ten_new_controllers']):
        raise ValueError('All four original GPU ranks and ten full downstream controllers must remain live')
    launched=read(ROOT/'reports/full-function-repair-comparison-CPU-v1-launch.json');research=read(RESEARCH/'plan.json')
    if (terminal(launched) or (RESEARCH/'failure.json').exists() or launched['pid']!=research['owner']['pid']
            or launched['ticks']!=research['owner']['ticks'] or research['profiles']!=['plain_repeat','semantic_repair','weighted_semantic_repair']):
        raise ValueError('Exact actual live frozen three-arm research owner required')
    protected=set()
    for name in subprocess.check_output(['rg','--files','reports','-g','plan.json','-g','*-queue.json'],cwd=ROOT,text=True).splitlines():
        raw=read(ROOT/name)
        if isinstance(raw,dict):protected.update(raw.get('source_files_sha256',{}))
    targets=[master_path,ROOT/'README.md',ROOT/'RESEARCH.md'];args=['rg','-l','-F','-g','*.json']
    for path in targets:args.extend(['-e','"'+str(path)+'"','-e','"'+str(path.relative_to(ROOT))+'"'])
    scanned=subprocess.run([*args,'reports'],cwd=ROOT,text=True,capture_output=True)
    if scanned.returncode not in [0,1]:raise ValueError(scanned.stderr)
    for name in scanned.stdout.splitlines():
        raw=json.loads((ROOT/name).read_text(),parse_int=str)
        if isinstance(raw,dict):protected.update(str(Path(p).resolve()) for p in raw.get('source_files_sha256',{}))
    for path in [master_path,ROOT/'README.md',ROOT/'RESEARCH.md']:
        if str(path) in protected:raise ValueError('Source-bound root summaries cannot be edited')
    prefix=[]
    for line in (runtime/'training-metrics.jsonl').read_bytes().splitlines(keepends=True):
        if not line.endswith(b'\n'):break
        prefix.append(line)
    rows=[json.loads(line) for line in prefix]
    if [r['step'] for r in rows]!=list(range(1,len(rows)+1)):raise ValueError('Every actual complete main source prefix required')
    capacity={};training={};paths=[Path(__file__),previous,RESEARCH/'plan.json',ROOT/'reports/full-function-repair-comparison-CPU-v1-launch.json',
        ROOT/'reports/full-function-repair-comparison-kernel-and-evaluation-v1-final-tests.log',runtime/'plan.json',
        ROOT/'reports/instance-lineage-recovery-v2/complete.json']
    for profile in research['profiles']:
        complete=RESEARCH/('capacity-'+profile)/'complete.json'
        if complete.exists():
            actual=read(complete)
            if not actual['complete'] or not actual['every_actor_Adam_metric_and_private_RNG_next_update_bit_exact']:
                raise ValueError('Every claimed completed full-size capacity profile must pass')
            capacity[profile]=actual;paths.append(complete)
        journal=RESEARCH/profile/'actual-training-metrics.jsonl'
        actual=[]
        if journal.exists():
            for line in journal.read_bytes().splitlines(keepends=True):
                if not line.endswith(b'\n'):break
                actual.append(json.loads(line))
        if [r['step'] for r in actual]!=list(range(1,len(actual)+1)):raise ValueError('Contiguous actual research training prefix required')
        training[profile]=dict(actual_complete_updates=len(actual),planned=306,
            actual_original_response_EOS_targets=sum(r['response_EOS_targets'] for r in actual))
    at=datetime.now(timezone.utc).isoformat();OUT.mkdir();(OUT/'captured-main-training-prefix.jsonl').write_bytes(b''.join(prefix))
    snapshot=read(RESEARCH/'progress.json');atomic_json(OUT/'captured-research-progress.json',snapshot)
    observation=dict(at=at,actual_main_owner=plan['owner'],actual_four_GPU_ranks=ranks,
        actual_complete_pretraining_updates=len(rows),planned_pretraining_updates=4447,
        actual_main_response_EOS_targets=int(sum(r['response_EOS_targets'] for r in rows)),
        GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
        actual_research_owner=launched,actual_full_size_capacity_profiles=capacity,actual_research_training=training,
        research_phase_snapshot=snapshot,actual_research_student_counts=research['actual_student_counts'],
        original_ten_downstream_controllers_live=True,persistent_free_bytes=shutil.disk_usage(ROOT).free,
        new_generated_model_quality_results_available=False,last_complete_local_code_correct=0,last_complete_local_code_tasks=48,
        previous_goal_turn_classification='progress',current_goal_turn_classification='progress')
    atomic_json(OUT/'runtime.json',observation)
    requirements['serious_full_model_training']['evidence']=(
        f'The unchanged four-L40 public/raw/function/retention epoch has {len(rows)} of4447 complete updates and '
        f'{observation["actual_main_response_EOS_targets"]} response/EOS targets in the captured durable report prefix. '
        'The fresh full-size CPU three-arm function study is actually running with its own immutable source plan, nice5 and four threads. '
        f'Completed actual capacity profiles: {list(capacity)}. Complete research training updates by arm: '
        f'{ {k:v["actual_complete_updates"] for k,v in training.items()} }. '
        'Every arm is fixed at306logical updates,4851full presentations and2287110identical original response/EOS targets. '
        'Actual full-size capacity/recovery must pass every profile before any research production update. No new generated correctness result exists yet.')
    requirements['any_code_correctness_and_efficiency']['evidence']=(
        'NOT ACHIEVED. Last completed whole-code-epoch functional accuracy remains0/48, ordinaryPython0/24. '
        'Separate source-train CE and trajectoryDPO both learn9/10optimized functions but remain0/4unoptimized controls; syntax/stopping gains do not establish semantic transfer. '
        'New actual CPU research compares plain repetition,791independently admitted semantic-repair views, and the same repair views with code4target weights. '
        'All826original train functions and all original gold targets/EOS are retained;15original validation functions never supply gradients. '
        'The new kernel passes exact full-objective gradient and private-state rejection/replay tests; the native evaluator preserves actual text/image actions and seals actual CPU BF16. '
        'Seven combined tests pass. The full-size study and all fixed native evaluations are still pending; no quality improvement, arbitrary-code coverage or optimal efficiency is claimed.')
    requirements['broad_independent_benchmarks']['evidence']=(
        'Every original nine endpoint,16961QA,3088AI2Dunder3conditions,600MBPP/1816cases,4642heldhuman and3429heldwordsunder3conditions remains in ten actual live queues. '
        'The new CPU study predeclares all15original source-pack validation functions at1024native actions plus the unchanged full48local coding tasks at512actions,56knowledge,16math,20OCR,15object and32photo checks for baseline and every final actor. '
        'All three arms finish training before those new baseline/final measurements, preventing result-dependent arm changes. These are development checks; source-pack validation is not certified all-history independent. '
        'No new generated assessment has completed and no original final benchmark population is reduced.')
    requirements['gpu_runtime']['evidence']=(
        'The same actual four original L40 ranks and production GPU lease remain intact; the main full-size capacity/private-next-step/cold proofs are already complete. '
        'New three-arm CPU BF16 research uses four threads at nice5, no visible CUDA devices, one shared frozen own base and an explicit training-only cast cache. '
        'Every research profile requires actual full-size source/backward/Adam/private-RNG/cold checks before production. Resource checks preserve26GiB disk reserve and budget32GiB CPU working allowance plus64GiB other-job headroom. '
        'Existing2thread cached/uncached resource timings are not claimed as measured4thread full-source throughput.')
    for value in requirements.values():value['status']='incomplete_for_requested_final_model'
    if len(requirements)!=10:raise ValueError('Every original ten final requirements must remain in scope')
    paths.extend([OUT/'runtime.json',OUT/'captured-main-training-prefix.jsonl',OUT/'captured-research-progress.json'])
    atomic_json(OUT/'review.json',dict(at=at,requirements=requirements,source_files_sha256={str(p):file_hash(p) for p in paths},
        previous_full_scope_review=str(previous.relative_to(ROOT)),previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True,original_full_scope_preserved=True,candidate_promoted=False,goal_achieved=False))
    (OUT/'previous-completion-audit.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:v['evidence'] for k,v in requirements.items()},latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),
        latest_actual_main_runtime=observation,all_ten_final_requirements_incomplete=True,current_goal_turn_classification='progress',candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Full-size three-arm coding study launched, **{at}**: the same four L40s have completed **{len(rows)}/4,447updates** '
        f'and **{observation["actual_main_response_EOS_targets"]:,} response/EOS targets** in this captured prefix. '
        'All ten original full human-feedback/RL/export/evaluation controllers remain live. '
        'A separate actual **999,706,442-parameter** CPU study now compares ordinary prompts, verified semantic-repair prompts, and repair prompts with extra executable-code target weight. '
        'Each arm retains all826original functions,791target-matched second views,3epochs,306updates and2,287,110response/EOS targets. '
        f'Its captured phase is **{snapshot["phase"]}**; completed full-size capacity profiles are **{list(capacity)}**. '
        'Seven kernel/evaluator tests pass. All15source-pack validation functions and unchanged48local coding tasks plus broad text/image checks are fixed before evaluation. '
        '**There is no new generated correctness result; the last whole-code-epoch result remains0/48functional.** '
        'The earlier9/10trained versus0/4unoptimized-control result is source learning without demonstrated transfer. '
        '**All ten final requirements remain incomplete.** '
        '[Live experiment plan](reports/full-function-repair-comparison-CPU-v1/plan.json), '
        '[full review](reports/full-function-repair-launch-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for path in [ROOT/'README.md',ROOT/'RESEARCH.md']:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),actual_main_updates=len(rows),actual_research_training=training,capacity_profiles=list(capacity),all_ten_incomplete=True)),flush=True)


if __name__=='__main__':main()
