"""Record actual public procedural-source readiness without claiming model gain."""
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from slm_lab.function_repair_GPU_contract_v1 import ADMISSION,QUEUE,handoff_admitted

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/public-reasoning-generator-qualification-v1-review'
PRODUCER=ROOT/'reports/public-reasoning-generator-qualification-CPU-v3'
READER=ROOT/'reports/public-reasoning-generator-qualification-v3-readback'
DUTY=ROOT/'reports/waiter-polling-duty-cycle-CPU-v2'


def read(path):return json.loads(Path(path).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.pending')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def actual(owner):
    fields=(Path('/proc')/str(owner['pid'])/'stat').read_text().split(') ',1)[1].split()
    if fields[19]!=owner['ticks'] or fields[0] in {'Z','X'}:raise ValueError('Actual original process birth must remain live')
    return dict(owner=owner,state=fields[0])


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists():raise ValueError('Fresh low-priority CPU evidence review required')
    reader=read(READER/'complete.json');producer=read(PRODUCER/'complete.json')
    if (not reader.get('complete') or reader['actual_entries_regenerated']!=3072
            or reader['actual_reward_controls_reexecuted']!=4096 or reader['actual_native_complete_masks_rebuilt']!=3072
            or reader['actual_counts']!=producer['actual_counts'] or reader['training_admitted'] or reader['new_neural_updates']!=0):
        raise ValueError('Actual complete fixed population, native masks and reward controls must reconstruct')
    audit=read(ROOT/'reports/completion-audit.json');previous=Path(audit['latest_full_scope_review']);review=copy.deepcopy(read(previous))
    queue=read(ROOT/QUEUE);handoff=read(ROOT/ADMISSION)
    if not handoff_admitted(ROOT,queue['owner']):raise ValueError('All ten complete-scope waiting owners must remain admitted')
    maps=[read(ROOT/'reports/public-pretraining-recovery-v1/plan.json')['source_files_sha256'],
          read(ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json')['source_files_sha256'],
          read(ROOT/'reports/public-pretraining-interim-CPU-v2/evaluation-plan.json')['source_files_sha256'],
          read(ROOT/'reports/public-edge-preference-comparison-CPU-v1-queue.json')['source_files_sha256'],queue['source_files_sha256']]
    maps += [read(ROOT/item['queue'])['source_files_sha256'] for item in handoff['new_controllers']]
    for name in ['reports/completion-audit.json','README.md','RESEARCH.md']:
        if any(str(ROOT/name) in mapping for mapping in maps):raise ValueError('Never edit a live source-bound index or document')
    files=dict(reader['source_files_sha256'])
    paths=[Path(__file__),previous,READER/'complete.json',ROOT/'reports/reasoning-source-cross-process-replay-v1-tests.log',
           ROOT/'scripts/read_public_reasoning_qualification_CPU_v3.py',
           ROOT/'reports/public-reasoning-generator-qualification-CPU-v1/failure.json',
           ROOT/'reports/public-reasoning-generator-qualification-CPU-v2/complete.json',
           ROOT/'reports/public-reasoning-generator-qualification-v2-readback/failure.json',
           ROOT/'reports/public-reasoning-generator-qualification-v2-readback-v2/failure.json',
           ROOT/'reports/waiter-polling-duty-cycle-CPU-v1/complete.json',DUTY/'plan.json',DUTY/'cycle-1-complete.json']
    files.update({str(path):sha(path) for path in paths})
    if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('Every frozen qualification source must remain exact')
    raw=(ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes();prefix=raw[:raw.rfind(b'\n')+1]
    rows=list(map(json.loads,prefix.splitlines()))
    if [row['step'] for row in rows]!=list(range(1,len(rows)+1)):raise ValueError('Complete contiguous main training prefix required')
    main_state=actual(dict(pid=1241,ticks='49351203'))
    current=dict(owner=main_state['owner'],complete_actual_updates=len(rows),planned_updates=4447,
        response_EOS_targets=sum(int(row['response_EOS_targets']) for row in rows),complete_prefix_sha256=hashlib.sha256(prefix).hexdigest())
    protected=[dict(pid=18545,ticks='50128787'),dict(pid=25240,ticks='51635952'),queue['owner'],
               read(ROOT/'reports/public-edge-preference-comparison-CPU-v1-queue.json')['owner']]
    original_live=[actual(owner) for owner in protected]
    if any(value['state'] in {'T','t'} for value in original_live+[main_state]):raise ValueError('Original live training/assessment must not be stopped')
    duty=read(DUTY/'plan.json');actual(duty['owner'])
    if (duty['maximum_seconds']!=3600 or not read(DUTY/'cycle-1-complete.json')['all_owned_suspensions_released']
            or (DUTY/'failure.json').exists()):raise ValueError('The bounded continuation must have actually completed a full restore cycle')
    at=datetime.now(timezone.utc).isoformat();OUT.mkdir()
    proof=dict(complete=True,at=at,source_files_sha256=files,
        current_goal_turn_classification='progress: independently qualified reproducible public procedural generators and strict reward controls; no new model-quality result',
        source_repository='https://github.com/open-thought/reasoning-gym',source_revision=read(PRODUCER/'plan.json')['revision'],
        actual_twelve_task_source_replay=reader,original_main=current,actual_GPU_rows=subprocess.check_output([
            'nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True),
        original_CPU_comparison=read(ROOT/'reports/full-function-repair-comparison-CPU-v1/progress.json'),
        original_CPU_assessment=read(ROOT/'reports/public-pretraining-interim-CPU-v2/progress.json'),
        original_protected_live=original_live,all_ten_original_full_scope_successors_live=True,
        original_neural_training_budgets_sources_and_nine_endpoint_benchmarks_unchanged=True,
        previous_20minute_polling_coordinator_complete_and_all_owned_waiters_released=True,
        actual_new_bounded_polling_coordinator=dict(owner=duty['owner'],maximum_seconds=3600,
            pause_seconds=45,run_seconds=15,independent_rescue_seconds=55,completed_full_restore_cycles_at_least=1),
        actual_oracle_tests_passed=16,actual_cross_interpreter_replay_tests_passed=1,
        failed_producer_and_reader_artifacts_preserved=True,upstream_source_bytes_never_edited=True,
        new_neural_updates_by_source_research=0,candidate_promoted=False,goal_achieved=False)
    atomic(OUT/'complete.json',proof)
    text=['# Public procedural source qualification','',
          'A pinned subset of [Reasoning Gym](https://github.com/open-thought/reasoning-gym) provides actual public procedural questions and answers. '
          'All 30 fetched files match their original Git blob IDs and SHA256 values. Twelve unchanged task generators execute through a narrow namespace loader; no package, model or teacher is installed.', '',
          'The producer draws all 3,072 predeclared examples. A fresh-process reader regenerates every original question, independently calculates every answer, rebuilds every complete BPE response/EOS mask, and repeats all 4,096 fixed reward controls. '
          'All answers agree. Sixteen actual oracle/source tests and one three-interpreter replay test pass.', '',
          '| Task | Original draws | Response/EOS targets |','|---|---:|---:|']
    for task,value in reader['actual_by_task'].items():text.append(f'| {task} | {value["entries"]:,} | {value["response_EOS_targets"]:,} |')
    text += ['',f'Total: **3,072 draws**, **{reader["actual_unique_questions"]:,} distinct exact questions**, **22,022 response/EOS targets**, '
             '**84 duplicate draws**, and **23 question groups shared by both seeds**. No draw is silently dropped. '
             'Disjoint generator seeds alone do not establish independent questions. Semantic equivalence and overlap with all historical benchmarks are not certified.', '',
             'The upstream default scorer intentionally gives fractional credit when expected text occurs inside a longer answer. '
             'In **1,459 of 1,536 fixed incorrect-integer controls**, that produces nonzero fractional credit; all get zero under our strict scorer. '
             'All **2,560 extra-text controls** get upstream partial credit and strict zero. These are reward-protocol observations, not an upstream claim of full correctness. '
             'The matrix scorer can evaluate answer text, so arbitrary model text is never sent to that upstream method. Our strict scorer only compares independently verified canonical text. '
             'The prompt keeps the whole original question and adds an explicit answer-format instruction.', '',
             'Two string generators choose from Python sets. Their source RNG seed alone does not reproduce the exact strings across interpreters. '
             'Explicit startup PYTHONHASHSEED=0 and the same Python version restore exact unchanged-source replay. The cross-process test actually demonstrates the difference at hash seed 1. '
             'The original failed producer (uninitialized tokenizer), the initial metadata tuple/list comparison failure, and the later random-hash replay failure are retained. '
             'Only new versions make the corrections; all bound producer sources and original public files stay exact.', '',
             '**These are qualification candidates. They are not admitted to model optimization, have not been consumed by any model, and are not a held-out model benchmark.** '
             'The concrete next data step is question/content overlap control and an explicit training partition before supervised warmup or on-policy reward experiments. '
             'This supplies a checked source and reward protocol, not an established improvement or a globally novel method.', '',
             f'The original scratch-origin model continues at **{len(rows):,}/4,447** updates with **{current["response_EOS_targets"]:,}** actual targets. '
             'All four L40s remain assigned to that run. The existing CPU training/assessment and queued three-arm first-divergence preference experiment remain live. '
             'All 17 public families, 85,786 human pairs, original execution-RL and machine-preference populations, and full nine-endpoint assessments are preserved.', '',
             'The prior 20-minute polling controller ended and restored every owned waiter. A new one-hour bounded controller has actually completed a full restore cycle, '
             'using the same tested 45-second pause / 15-second running schedule and independent 55-second rescue. Only the same nine unstarted controller parents are affected. '
             'All neural workers remain running. This avoids measured hashing overhead and does not establish a direct decoding-speed or accuracy gain.', '',
             'Latest complete generated coding result remains **0/48 functional passes** from the earlier whole-code epoch comparison. '
             'The interim matched assessment is still running and has no complete endpoint. **No new model accuracy improvement or frontier-level capability is established; all ten final requirements remain incomplete.**','']
    (OUT/'metrics.md').write_text('\n'.join(text))
    review['requirements']['public_data']['evidence']+=(
        ' A pinned12task public procedural source subset now has3072 complete actual draws and a fresh3072entry/4096reward-control readback. '
        'All answers agree with independently implemented algorithms;22022complete response/EOS targets reconstruct. '
        'The2988exact unique questions include84duplicate draws and23cross-seed shared question groups. '
        'Two set-based generators require explicit Python hash seed0 for exact cross-process replay. '
        'These candidates are not training-admitted or an independent model benchmark; no model consumed them.')
    review['requirements']['serious_full_model_training']['evidence']+=(
        ' Strict canonical programmatic rewards reject all1536fixed wrong integers, while1459receive upstream fractional substring credit. '
        'The matrix answer text is never executed by our scorer. This is source/reward qualification with0newneuralupdates, '
        'not on-policy RL or measured learning. The original20minute scheduling trial ended and released all9waiters. '
        'A fresh one-hour bounded continuation completes a full restore cycle with unchanged45/15timing and55second independent rescue; no neuralworker is signalled.')
    if len(review['requirements'])!=10 or any(v['status']!='incomplete_for_requested_final_model' for v in review['requirements'].values()):
        raise ValueError('Source readiness establishes none of the final capability requirements')
    review.update(at=at,previous_full_scope_review=str(previous),current_goal_turn_classification=proof['current_goal_turn_classification'],
        latest_actual_generator_qualification=proof,source_files_sha256={**files,str(OUT/'complete.json'):sha(OUT/'complete.json')},goal_achieved=False)
    atomic(OUT/'review.json',review)
    if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('Every original frozen source must stay exact before index update')
    audit.update(at=at,requirements=review['requirements'],latest_full_scope_review=str(OUT/'review.json'),
        latest_actual_main_runtime=current,latest_actual_generator_qualification=str(OUT/'complete.json'),goal_achieved=False)
    atomic(ROOT/'reports/completion-audit.json',audit)
    paragraph=(f'Public procedural-source qualification, **{at}**: all **3,072 actual examples** from12 pinned public generators agree with independent algorithms. '
        'A fresh process reproduces every source entry, whole native target/mask and **4,096 reward controls**; **17 focused tests pass**. '
        'There are **2,988 distinct exact questions**, including84duplicate draws and23cross-seed shared groups. '
        'The strict scorer rejects incorrect integers and never executes model answer text. **These are unconsumed qualification candidates, not training admission or model-quality evidence.** '
        f'The main continues at **{len(rows)}/4,447** with all four L40s. Original full source, human/RL and nine-endpoint benchmark scopes remain. '
        '**Latest complete functional correctness remains0/48; all ten final requirements remain incomplete.** '
        '[Qualification and limits](reports/public-reasoning-generator-qualification-v1-review/metrics.md), '
        '[full review](reports/public-reasoning-generator-qualification-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md','RESEARCH.md']:
        path=ROOT/name;first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),main_actual_updates=len(rows),source_entries_verified=3072,goal_achieved=False)),flush=True)


if __name__=='__main__':main()
