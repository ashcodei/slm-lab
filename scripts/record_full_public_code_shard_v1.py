"""Review the live complete-shard audit and measured CPU cache finding."""
import copy
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

import pyarrow.parquet as pq

from record_public_code_instruction_screen_v1 import read,sha,atomic,live
from slm_lab.function_repair_GPU_contract_v1 import ADMISSION,QUEUE,handoff_admitted
from slm_lab.public_code_instruction_screen_v1 import parse_whole,UnsupportedExample
from slm_lab.public_synthetic_code_screen_policy_v1 import published_all_pass,body_fingerprint

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/full-public-code-shard-qualification-CPU-v1'
CACHE=ROOT/'reports/CPU-autocast-cache-inspection-v1/complete.json'
OBSERVATION=ROOT/'reports/full-shard-original-jobs-observation-CPU-v1.json'
OUT=ROOT/'reports/full-public-code-shard-readiness-v1-review'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists():raise ValueError('Fresh low-priority CPU review required')
    plan=read(SOURCE/'plan.json');progress=read(SOURCE/'progress.json');cache=read(CACHE);observation=read(OBSERVATION)
    source_domains=Counter();provider_pass=0;source_rows=0
    for batch in pq.ParquetFile(ROOT/'data/public-code-instruction-candidate-v1/train-00000-of-00050.parquet').iter_batches(
            batch_size=1024,columns=['unit_tests','tests_execution_status','average_test_score','domain'],use_threads=False):
        for row in batch.to_pylist():
            source_rows+=1;source_domains[row['domain']]+=1;provider_pass+=published_all_pass(row)
    if (plan['exact_whole_shard_rows']!=100000 or progress['original_rows_loaded']!=100000 or source_rows!=100000
            or provider_pass!=34733 or dict(source_domains)!={'generic':90871,'algorithmic':9129} or plan['minimum_free_bytes']!=40*1024**3
            or progress['training_admitted'] or (SOURCE/'failure.json').exists()
            or not cache.get('complete') or cache['actual_shared_context_cast_counts']!=[15,0,0]
            or cache['actual_separate_context_cast_counts']!=[15,15] or not cache['shared_context_cached_weight_reuse_observed']
            or not observation.get('complete') or observation['wall_seconds']<45 or observation['actual_main_update_delta']<1):
        raise ValueError('Actual full-source reconstruction, live work and measured shared-context cache reuse required')
    producer=live(plan['owner'])
    environment=(Path('/proc')/str(plan['owner']['pid'])/'environ').read_bytes().split(b'\0')
    stat=(Path('/proc')/str(plan['owner']['pid'])/'stat').read_text().split(') ',1)[1].split()
    if b'CUDA_VISIBLE_DEVICES=' not in environment or int(stat[16])!=19:raise ValueError('Actual CUDA-hidden nice19 source owner required')
    launch=read(ROOT/'reports/full-public-code-shard-qualification-CPU-v1-launch.json')
    if launch['owner']!=plan['owner'] or launch['producer_sha256']!=sha(launch['producer']):raise ValueError('The actual launched producer bytes and birth must remain exact')
    with (ROOT/'data/public-code-instruction-candidate-v1/first1024-original-entries.jsonl').open() as stream:rows=[json.loads(x)['original_entry'] for x in stream]
    with (ROOT/'reports/public-code-instruction-readback-CPU-v1/all1024-readback-observations.jsonl').open() as stream:prior=list(map(json.loads,stream))
    bad=body_fingerprint(parse_whole(rows[30])['program']);pilot=dict(provider_exclusion=0,whole_test_abstention=0,known_bad_AST_veto=0,prequalified=0,prior_fresh_pass_reusable=0,requires_new_execution=0)
    for position,row in enumerate(rows):
        if not published_all_pass(row):pilot['provider_exclusion']+=1;continue
        try:parsed=parse_whole(row)
        except UnsupportedExample:pilot['whole_test_abstention']+=1;continue
        if body_fingerprint(parsed['program'])==bad:pilot['known_bad_AST_veto']+=1;continue
        pilot['prequalified']+=1
        key='prior_fresh_pass_reusable' if prior[position].get('fresh_whole_program_execution',{}).get('all_published_tests_passed') else 'requires_new_execution'
        pilot[key]+=1
    if pilot!={'provider_exclusion':535,'whole_test_abstention':145,'known_bad_AST_veto':1,'prequalified':343,'prior_fresh_pass_reusable':340,'requires_new_execution':3}:
        raise ValueError('The actual fixed whole-pilot policy outcomes must reconstruct without silently assuming metadata equals execution')
    audit=read(ROOT/'reports/completion-audit.json');previous=Path(audit['latest_full_scope_review']);review=copy.deepcopy(read(previous))
    gpu=read(ROOT/QUEUE);handoff=read(ROOT/ADMISSION)
    if not handoff_admitted(ROOT,gpu['owner']):raise ValueError('All ten original full-scope successors must remain admitted')
    paths=[ROOT/'reports/public-pretraining-recovery-v1/plan.json',ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json',
        ROOT/'reports/public-pretraining-interim-CPU-v2/evaluation-plan.json',ROOT/'reports/public-edge-preference-comparison-CPU-v1-queue.json',
        ROOT/'reports/public-reasoning-warmup-CPU-v1-queue.json',ROOT/QUEUE,*[ROOT/item['queue'] for item in handoff['new_controllers']]]
    plans=[read(p) for p in paths];maps=[p['source_files_sha256'] for p in plans]+[plan['source_files_sha256'],cache['source_files_sha256']]
    for name in ['reports/completion-audit.json','README.md','RESEARCH.md']:
        if any(str(ROOT/name) in m for m in maps):raise ValueError('Never edit an index or document bound by active work')
    files={**plan['source_files_sha256'],**cache['source_files_sha256']}
    for p in [Path(__file__),previous,CACHE,OBSERVATION,SOURCE/'plan.json',ROOT/'scripts/observe_full_shard_and_original_jobs_CPU_v1.py',
            ROOT/'reports/full-public-code-shard-qualification-CPU-v1-launch.json',*paths]:files[str(p)]=sha(p)
    if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every fixed source, policy, test and observation must remain exact')
    originals=[live(p['owner']) for p in plans[:6]]
    if any(item['state'] in {'T','t'} for item in originals[:3]):raise ValueError('No original neural worker may be stopped')
    raw=(ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes();prefix=raw[:raw.rfind(b'\n')+1]
    metrics=[json.loads(x) for x in prefix.split(b'\n') if x]
    if [r['step'] for r in metrics]!=list(range(1,len(metrics)+1)):raise ValueError('All contiguous actual main updates required')
    current=dict(owner=plans[0]['owner'],complete_actual_updates=len(metrics),planned_updates=4447,
        response_EOS_targets=sum(int(r['response_EOS_targets']) for r in metrics))
    at=datetime.now(timezone.utc).isoformat();OUT.mkdir();current_progress=read(SOURCE/'progress.json')
    proof=dict(complete=True,at=at,source_files_sha256=files,
        previous_goal_turn_classification='progress:7small own adapter-folding/recycling controls and44to16linear calls; production adoption pending',
        current_goal_turn_classification='progress: full100000row public shard reconstructed and live single-thread CPU qualification; measured CPU cache hypothesis ruled out in the tested path',
        actual_source_owner=producer,actual_source_progress=current_progress,actual_source_domain_counts=dict(source_domains),
        original_shard_rows=100000,provider_complete_all_pass_rows=34733,actual_new_policy_and_disk_wait_tests_passed=5,
        actual_whole1024pilot_policy=pilot,actual_CPU_autocast_cache_control=cache,
        actual_verified45second_observation=observation,actual_original_main=current,actual_original_live_owners=originals,
        actual_GPU_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True),
        source_teacher='Qwen2.5-Coder-32B-Instruct',student_weights_remain_own_scratch_origin=True,
        external_model_weights_or_live_teacher_API_used=False,all_new_data_remains_unadmitted=True,
        future_candidate_SFT_is_explicit_weak_synthetic_supervision_not_independent_RL_truth=True,
        full_shard_qualification_readback_roles_packs_and_model_learning_pending=True,
        CPU_cache_control_small_only_not_full_size_throughput_or_accuracy=True,
        no_live_neural_worker_signal_source_priority_affinity_or_numeric_change=True,
        all_original17family_human_RL_and_nine_endpoint_scopes_preserved=True,
        new_production_optimizer_updates_from_this_work=0,latest_complete_local_functional_correctness='0/48',
        new_model_quality_gain_established=False,candidate_promoted=False,goal_achieved=False)
    atomic(OUT/'complete.json',proof)
    text=(f'Complete-shard source audit active, {at}. The exact previously downloaded public shard is reconstructed as100,000original rows: '
        '90,871generic and9,129algorithmic. Provider metadata marks34,733as all-test-pass; this is a prefilter, not our correctness count. '
        f'The actual CUDA-hidden nice19owner is PID{plan["owner"]["pid"]}, birth{plan["owner"]["ticks"]}. '
        'It screens all source positions against31,349protected inputs, preserves whole original programs/tests/native responses, '
        'reuses only exact source-bound fresh pilot executions, and executes eligible remaining programs in existing OS confinement. '
        'The known mathematically wrong source and its matching AST are explicitly vetoed.\n\n'
        'Five actual policy/resource controls pass. Rebuilding the first1,024positions yields535provider exclusions,145whole-test abstentions, '
        'one known bad AST veto and343prequalified entries. Only340already have reusable passed fresh execution; the other3require new execution. '
        'No metadata flag is promoted to an independent correctness label.\n\n'
        'Answers and published tests are Qwen-generated public synthetic data. Any future use is explicit weak supervised learning; '
        'they are prohibited as an independent RL reward oracle. No external weights or live teacher API are used. '
        'All new data remains unadmitted pending the complete audit, fresh full readback, sealed roles, complete native packs and later model comparisons. '
        'This one Python instruction shard is not coverage of all code, all languages or the full five-million-row dataset.\n\n'
        'The producer keeps a40GiBdisk reserve, a tested bounded wait for transient atomic checkpoint saves,16GiBresident allowance and64GiBother-process headroom. '
        'No new neural training job or GPU lease is started.\n\n'
        'An actual small native-path CPU control records15weight casts during initial BF16 prefill and0/0on subsequent decode calls in one shared context. '
        'Deliberately ending that context causes15/15casts, with bit-exact unchanged logits. The active evaluator source already uses the shared context, KV cache and rotary cache. '
        'This does not justify changing its precision/casting path; full-size live throughput is not profiled by that small control.\n\n'
        f'A verified45secondwindow keeps every exact original/source birth live and records{observation["actual_main_update_delta"]}main updates '
        f'and{observation["actual_main_target_delta"]:,}targets, while the data worker consumes actual CPU time. '
        f'The main is now{len(metrics):,}/4,447 with{current["response_EOS_targets"]:,}targets. '
        'Latest completed local coding accuracy remains0/48 on an older checkpoint. Newer matched evaluation and all final benchmarks remain pending. '
        'All original source/human/RL/nine-endpoint scopes remain intact. All ten final requirements remain incomplete.\n')
    (OUT/'metrics.md').write_text(text)
    if len(review['requirements'])!=10 or any(v['status']!='incomplete_for_requested_final_model' for v in review['requirements'].values()):
        raise ValueError('Source qualification readiness establishes none of the requested final capabilities')
    review['current_actual_runtime']['main']=current
    review.update(at=at,previous_full_scope_review=str(previous),previous_goal_turn_classification=proof['previous_goal_turn_classification'],
        current_goal_turn_classification=proof['current_goal_turn_classification'],latest_actual_full_public_code_shard=proof,
        source_files_sha256={**files,str(OUT/'complete.json'):sha(OUT/'complete.json')},goal_achieved=False)
    atomic(OUT/'review.json',review)
    if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every pinned byte must remain exact before index update')
    audit.update(at=at,requirements=review['requirements'],latest_full_scope_review=str(OUT/'review.json'),latest_actual_main_runtime=current,
        latest_actual_full_public_code_shard=str(OUT/'complete.json'),goal_achieved=False)
    atomic(ROOT/'reports/completion-audit.json',audit)
    paragraph=(f'Full public coding shard audit active, **{at}**: all**100,000**original rows are reconstructed and an actual CUDA-hidden nice19owner screens them. '
        'Provider34,733pass flags are a prefilter; whole local execution, native fit, protected input screening and the known semantic veto govern candidates. '
        '**All new synthetic data remains unadmitted; no model gain is claimed.** Five new policy/resource controls pass. '
        'A small actual cache control confirms15initial weight casts followed by0/0shared-context decode casts, so the existing evaluator is preserved. '
        f'The original main continues at**{len(metrics)}/4,447**. **Latest complete coding correctness remains0/48 on an older checkpoint; all ten final requirements remain incomplete.** '
        '[Readiness and actual observations](reports/full-public-code-shard-readiness-v1-review/metrics.md), '
        '[full review](reports/full-public-code-shard-readiness-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md','RESEARCH.md']:
        path=ROOT/name;first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(complete=True,actual_source_owner=plan['owner'],actual_main_updates=len(metrics),training_admitted=False,goal_achieved=False)))


if __name__=='__main__':main()
