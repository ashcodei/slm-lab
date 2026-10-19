"""Execute bounded reference/noop probes on every existing admitted train function."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/bounded-code-probes-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Probe preparation must hide every GPU')
    if REPORTS.exists():raise ValueError('Preserve earlier complete/partial preparations')
    from slm_lab.bounded_code_probes_v1 import qualify,evaluate
    from slm_lab.causal_code_reward_catalog_v1 import CausalPublicCodeReward
    from slm_lab.causal_code_reward_v2 import pass_ablation
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    tests=ROOT/'reports/bounded-code-probes-v1-tests-v3.log'
    if '5 passed' not in tests.read_text():raise ValueError('Actual source/role/side-effect/docstring/coverage regression tests required')
    catalog=CausalPublicCodeReward(ROOT);started=time.monotonic()
    files=[Path(__file__),ROOT/'src/slm_lab/bounded_code_probes_v1.py',ROOT/'tests/test_bounded_code_probes_v1.py',tests,
        ROOT/'src/slm_lab/documented_doctest_v1.py',ROOT/'scripts/documented_doctest_raw_harness_v1.py',
        ROOT/'src/slm_lab/standalone_program_v1.py',ROOT/'src/slm_lab/causal_code_reward_v2.py',
        ROOT/'reports/causal-code-rewards-v2/admission.json',ROOT/'reports/causal-code-rewards-v2/observations.jsonl',
        ROOT/'data/documented-doctest-verified-v1/admission.json',ROOT/'data/documented-doctest-stage-v1/candidates.jsonl',
        ROOT/'reports/documented-doctest-verified-v1/observations.jsonl']
    bindings={str(p):file_hash(p) for p in files};guard=FileBindings(bindings);REPORTS.mkdir()
    atomic_json(REPORTS/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bindings,
        existing_admitted_original_functions=len(catalog.candidates),maximum_proposals_per_function=16,workers=2,seconds_per_worker_suite=5,
        original_publisher_examples_never_rewritten=True,new_examples_are_machine_generated_reference_agreement_only=True,
        reference_accepted_execution_not_formal_input_domain_certificate=True,independent_subset_replay_and_admission_still_required=True,
        new_raw_functions=0,new_human_labels=0,neural_updates=0,goal_achieved=False))
    def one(key):
        c=catalog.candidates[key];q=qualify(c);row=dict(source_id=key,qualification=q)
        if q['available']:
            correct=evaluate(c,catalog.references[key],catalog.controls[key],q,c['program'],origin='test_fixture')
            noop=evaluate(c,catalog.references[key],catalog.controls[key],q,pass_ablation(c),origin='test_fixture')
            if correct['reward']!=1. or noop['reward']!=0.:
                # Preserve the discovery but do not admit observations that
                # fail to reproduce after rejected proposals are removed.
                row['subset_replay_qualification']=dict(correct_control=correct,no_op_control=noop)
                row['qualified_for_independent_reader']=False
            else:row.update(correct_source_control=correct,no_op_control=noop,qualified_for_independent_reader=True)
        else:row['qualified_for_independent_reader']=False
        return row
    rows=[]
    with (REPORTS/'observations.jsonl').open('x') as f,ThreadPoolExecutor(max_workers=2) as pool:
        for row in pool.map(one,sorted(catalog.candidates)):
            rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            if len(rows)%25==0:atomic_json(REPORTS/'progress.json',dict(functions_processed=len(rows),total_functions=len(catalog.candidates),
                functions_with_qualified_extra_probes=sum(r['qualified_for_independent_reader'] for r in rows),neural_updates=0,goal_achieved=False))
    guard.check(full=True);good=[r for r in rows if r['qualified_for_independent_reader']]
    result=dict(at=datetime.now(timezone.utc).isoformat(),original_admitted_functions=len(rows),functions_with_qualified_extra_probes=len(good),
        functions_without_qualified_extra_probes=len(rows)-len(good),new_machine_probe_observations=sum(len(r['qualification']['accepted']) for r in good),
        source_correct_reward_one_controls=len(good),no_op_reward_zero_controls=len(good),source_files_sha256=bindings,
        observations_sha256=file_hash(REPORTS/'observations.jsonl'),seconds=time.monotonic()-started,
        independent_reader_and_new_reward_training_integration_pending=True,new_reward_admitted=False,
        original_715_publisher_examples_retained=True,no_held_benchmark_used=True,no_new_raw_function_or_human_judgment=True,
        neural_updates=0,quality_gain_proven=False,goal_achieved=False)
    atomic_json(REPORTS/'prepared.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_files_sha256'}),flush=True)


if __name__=='__main__':main()
