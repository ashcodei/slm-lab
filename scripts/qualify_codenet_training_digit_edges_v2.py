"""Qualify the targeted source probe before any execution-preference admission."""
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.native_code import toolchain
from slm_lab.standalone_program_v1 import compile_cpp_program,run_bounded

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'reports/codenet-training-digit-edges-v1'
    output=folder/'qualified-source-counterexamples-v2.json'
    if output.exists():raise ValueError('Preserve prior qualification evidence')
    plan_path,result_path=folder/'plan.json',folder/'results.json'
    plan=json.loads(plan_path.read_text());result=json.loads(result_path.read_text())
    assert result['plan_sha256']==file_hash(plan_path) and result['original_train_only'] and result['neural_updates']==0
    guard=FileBindings({**plan['source_files_sha256'],str(plan_path):file_hash(plan_path),str(result_path):file_hash(result_path),
        str(Path(__file__).resolve()):file_hash(__file__)})
    original={}
    with gzip.open(ROOT/'data/codenet-durable-pack-v2/original-rows.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line)
            if row['source_problem_id']=='p00002' and row['language'] in {'C++','Python'}:
                assert row['split']=='train' and row['group_id']=='p00002'
                original[row['id']]=row
    assert set(original)=={r['id'] for r in result['records']}
    correct=[];wrong=[];qualified=[];newline_probes=[]
    for record in result['records']:
        source=original[record['id']]
        assert source['original_source_body_sha256']==record['original_source_body_sha256']
        assert hashlib.sha256(source['response'].encode()).hexdigest()==record['original_source_body_sha256']
        if not record['cases']:
            qualified.append(dict(id=record['id'],reason=record['qualification']));continue
        # Recompute every finite original observation. Failed executions are not
        # wrong-output labels. Only the generated source-boundary case admits a
        # counterexample here, independently of HTML sample line termination.
        for case in record['cases']:
            passed=case['status']=='completed' and case['stdout'].split()==case['expected'].split()
            assert passed==case['passed']
        edges=[c for c in record['cases'] if c['kind']=='derived_source_boundaries']
        assert len(edges)==1 and edges[0]['input'].endswith('\n')
        edge=edges[0]
        if edge['status']=='completed' and not edge['passed']:
            observed=edge['stdout'].split();expected=edge['expected'].split()
            assert expected[0]=='1' and observed[0]=='0' and expected[1:]==observed[1:]
            wrong.append(dict(id=record['id'],reason='Completed0+0output0instead_of1_digit',
                source_body_sha256=record['original_source_body_sha256'],input='0 0\n',
                complete_observed_batch_stdout=edge['stdout'],complete_expected_batch_stdout=edge['expected'],
                zero_based_failing_line=0,original_publisher_samples_passed=record['all_original_publisher_samples_passed']))
        elif all(c['passed'] for c in record['cases']):
            correct.append(record['id'])
        else:
            # The original HTML extraction does not preserve a final LF. A
            # source that consumes character lines may require its restoration;
            # the raw observation remains preserved, never silently repaired.
            assert record['language']=='C++' and edge['passed']
            tools=toolchain();probes=[]
            with tempfile.TemporaryDirectory(prefix='slm-source-newline-qualification-') as work:
                constraints,compilation=compile_cpp_program(source['response'].encode(),work,tools)
                assert constraints is not None and compilation['status']=='completed'
                for case in record['cases']:
                    if case['kind']!='original_publisher_sample':continue
                    canonical=case['input'].rstrip('\n')+'\n'
                    observed=run_bounded(constraints,canonical.encode())
                    actual=observed['stdout'].decode(errors='replace')
                    assert observed['status']=='completed' and actual.split()==case['expected'].split()
                    probes.append(dict(input=canonical,original_extracted_input=case['input'],stdout=actual,
                        expected=case['expected'],status=observed['status'],source_unmodified=True))
            newline_probes.append(dict(id=record['id'],reason='Raw extracted sample omitted final newline; canonical line-terminated input passes.',probes=probes))
            qualified.append(dict(id=record['id'],reason='sample_final_newline_sensitive; not a verified algorithm-error label'))
    guard.check(full=True)
    assert len(wrong)==2 and len(correct)==4 and len(qualified)==2
    chosen=next(i for i in correct if original[i]['language']=='C++')
    proposed_pairs=[dict(split='train',group_id='p00002',prompt_source_id=chosen,
        chosen_original_source_id=chosen,rejected_original_source_id=item['id'],
        chosen_body_sha256=original[chosen]['original_source_body_sha256'],rejected_body_sha256=item['source_body_sha256'],
        label_scope='Original chosen source passes original sample and16derived boundary lines; rejected source has a completed wrong0+0output. Finite execution evidence only, not human preference or whole-program correctness.') for item in wrong]
    atomic_json(output,dict(completed_at=datetime.now(timezone.utc).isoformat(),passed=True,
        source_plan_sha256=file_hash(plan_path),raw_results_sha256=file_hash(result_path),
        source_files_sha256={str(Path(__file__).resolve()):file_hash(__file__),**plan['source_files_sha256']},
        raw_v1_counterexample_count=3,qualified_completed_boundary_counterexamples=wrong,
        all_targeted_literal_samples_and_edge_cases_pass=correct,other_qualifications=qualified,
        canonical_newline_followup=newline_probes,proposed_original_source_pairs=proposed_pairs,
        training_data_or_preferences_admitted=False,model_updates=0,human_feedback_labels=False,goal_achieved=False,
        scope='Two complete original training programs fail one unambiguous decimal0digit boundary. One extra raw sample mismatch is an input-final-newline qualification, not an algorithm-error label. All original observations preserved. Two candidate same-question finite-execution preference pairs are proposed only; no source population is relabeled, dropped or trained here.'))
    print(json.dumps(dict(verified_zero_digit_counterexamples=len(wrong),literal_and_boundary_pass=len(correct),
        other_qualifications=qualified,proposed_unadmitted_source_pairs=len(proposed_pairs)),indent=2))


if __name__=='__main__':main()
