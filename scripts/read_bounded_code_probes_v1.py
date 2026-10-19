"""Independently reconstruct and replay each qualified new reference probe."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import ast
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/bounded-code-probes-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU independent reader must hide all GPUs')
    if (REPORTS/'independent-reader.json').exists() or (REPORTS/'admission.json').exists():raise ValueError('Preserve earlier readback/admission')
    from slm_lab.bounded_code_probes_v1 import execute_suite,observation_expression
    from slm_lab.causal_code_reward_catalog_v1 import CausalPublicCodeReward
    from slm_lab.causal_code_reward_v2 import pass_ablation
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.documented_doctest_v1 import contract,compare,sha
    prepared=json.loads((REPORTS/'prepared.json').read_text());catalog=CausalPublicCodeReward(ROOT)
    if file_hash(REPORTS/'observations.jsonl')!=prepared['observations_sha256']:raise ValueError('Actual raw discovery/control ledger changed')
    rows=list(map(json.loads,(REPORTS/'observations.jsonl').open()))
    if [r['source_id'] for r in rows]!=sorted(catalog.candidates):raise ValueError('Every existing173train source requires an explicit disposition')
    bindings=dict(prepared['source_files_sha256'])
    for p in [Path(__file__),REPORTS/'prepared.json',REPORTS/'observations.jsonl']:bindings[str(p)]=file_hash(p)
    guard=FileBindings(bindings);started=time.monotonic();jobs=[];extra=0
    for row in rows:
        c=catalog.candidates[row['source_id']];q=row['qualification'];examples=contract(c);n=len(examples)
        if q['source_id']!=row['source_id'] or q['original_program_sha256']!=c['program_sha256']:raise ValueError('Exact original train source identity required')
        if not row['qualified_for_independent_reader']:continue
        reference=q['raw_reference_discovery'];noop=q['raw_noop_discovery'];proposed=q['proposed']
        if (not q['available'] or not reference['complete'] or not noop['complete'] or reference['actual_program_sha256']!=c['program_sha256']
                or noop['actual_program_sha256']!=sha(pass_ablation(c)) or len(reference['raw_results'])!=n+len(proposed)
                or len(noop['raw_results'])!=len(reference['raw_results'])
                or not all(compare(e,r) for e,r in zip(examples,reference['raw_results'][:n]))):raise ValueError('Every actual original/reference/noop discovery output must reconstruct')
        accepted=[]
        for i,p in enumerate(proposed):
            if p['observation_expression']!=observation_expression(c['function'],p['call_expression']):raise ValueError('Exact bounded input and post-call observation required')
            ref=reference['raw_results'][n+i];base=noop['raw_results'][n+i]
            if ref['exception_only'] is not None or ref['entry_body_calls']==0 or len(ref['stdout'])>8192:continue
            try:ast.literal_eval(ref['stdout'].strip())
            except (ValueError,SyntaxError,MemoryError,RecursionError):continue
            if base['exception_only'] is None and base['stdout']==ref['stdout']:continue
            accepted.append(i)
        if accepted!=[p['proposal_position'] for p in q['accepted']]:raise ValueError('Every and only informative accepted literal observation must be retained')
        for p in q['accepted']:
            if p['expected_observation_stdout']!=reference['raw_results'][n+p['proposal_position']]['stdout']:
                raise ValueError('Expected observations must come from actual confined original-source execution')
        if row['correct_source_control']['reward']!=1 or row['no_op_control']['reward']!=0:raise ValueError('Actual producer source/noop controls must be1/0')
        extra+=len(accepted);jobs.append((c,q))
    def replay(item):
        c,q=item;probes=q['accepted'];ref=execute_suite(c,probes);noop=execute_suite(c,probes,program=pass_ablation(c),origin='machine_noop');n=len(contract(c))
        if not ref['complete'] or not all(ref['original_example_passes']) or not noop['complete']:raise ValueError('Every fresh bounded subset/reference/noop must finish')
        ref_flags=[];noop_flags=[]
        for p,r,b in zip(probes,ref['raw_results'][n:],noop['raw_results'][n:]):
            if r['exception_only'] is not None or r['stdout']!=p['expected_observation_stdout'] or r['entry_body_calls']==0:
                raise ValueError('Every actual new original-source observation must reproduce independently')
            ref_flags.append(True);noop_flags.append(b['exception_only'] is None and b['stdout']==p['expected_observation_stdout'])
        if any(noop_flags):raise ValueError('No accepted extra observation may be passed by its no-op counterfactual')
        return dict(source_id=c['source']['id'],independently_reexecuted_added_probes=len(probes),
            reference_probe_passes=ref_flags,no_op_probe_passes=noop_flags,raw_reference_execution=ref,raw_no_op_execution=noop)
    replay_path=REPORTS/'independent-probe-replays.jsonl'
    if replay_path.exists():raise ValueError('Preserve earlier partial replays')
    with replay_path.open('x') as f,ThreadPoolExecutor(max_workers=2) as pool:
        for actual in pool.map(replay,jobs):f.write(json.dumps(actual,ensure_ascii=False)+'\n');f.flush()
    if len(jobs)!=prepared['functions_with_qualified_extra_probes'] or extra!=prepared['new_machine_probe_observations']:
        raise ValueError('Whole source/probe populations must reconstruct')
    guard.check(full=True)
    result=dict(at=datetime.now(timezone.utc).isoformat(),all_original_source_dispositions_checked=len(rows),
        original_functions_with_independently_reexecuted_extra_probes=len(jobs),actual_added_probe_observations_reexecuted=extra,
        all_original_reference_outputs_reproduced=True,all_added_probe_no_ops_failed=True,
        original_715_publisher_examples_unchanged=True,machine_added_reference_agreement_not_new_publisher_or_human_examples=True,
        reference_agreement_not_formal_input_domain_or_general_correctness_certificate=True,
        raw_replays_sha256=file_hash(replay_path),source_files_sha256=bindings,seconds=time.monotonic()-started,
        neural_updates=0,quality_gain_proven=False,current_production_reward_not_modified=True,goal_achieved=False)
    atomic_json(REPORTS/'independent-reader.json',result)
    atomic_json(REPORTS/'admission.json',dict(reward_ledger_sha256=file_hash(REPORTS/'observations.jsonl'),
        independent_reader_sha256=file_hash(REPORTS/'independent-reader.json'),source_files_sha256=prepared['source_files_sha256'],
        bounded_reference_probe_scoring_admitted=True,current_production_reward_not_modified=True,new_neural_training_started=False,goal_achieved=False))
    print(json.dumps({k:v for k,v in result.items() if k!='source_files_sha256'}),flush=True)


if __name__=='__main__':main()
