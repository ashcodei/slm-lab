"""Independently reconcile the four repaired suites and preserve all v1 views."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.execution_code_feedback_v1 import sequence
from slm_lab.model import ModelConfig
from slm_lab.python_execution_feedback_v1 import mutations
from slm_lab.python_execution_feedback_v2 import PublishedPythonSuitesV2
from slm_lab.training import Encoder

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'data/python-execution-feedback-v1';DATA=ROOT/'data/python-execution-feedback-v2'
REPORTS=ROOT/'reports/python-execution-feedback-v2';SOURCE=ROOT/'data/exercism-support-context-v3'
TOKEN=ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json'
PREFIX='SLM_PYTHON_PUBLISHER_INVENTORY_V2\t'
TARGETS={'electric-bill','meltdown-mitigation','diffie-hellman','error-handling'}


def raw_measure(observed,reference=None):
    run=observed['execution']
    if run['returncode']!=0 or run['limit'] is not None:return 'runtime_or_environment_qualification',None
    raw=observed['raw_inventory_output']
    if hashlib.sha256(raw.encode()).hexdigest()!=observed['report_sha256']:raise ValueError('Raw v2 byte/hash mismatch')
    lines=[x[len(PREFIX):] for x in raw.splitlines() if x.startswith(PREFIX)]
    if len(lines)!=1:return 'incomplete_test_inventory_qualification',None
    r=json.loads(lines[0]);parents=r['registered'];starts=r['started'];events=r['events']
    if r['format']!='own-complete-python-publisher-events-v2' or r['framework']!='pytest':raise ValueError('Wrong raw v2 framework')
    if len(parents)!=len(set(parents)) or len(starts)!=len(set(starts)):raise ValueError('Repeated raw parent identity')
    primary={};children={};contexts={};counters={}
    for e in events:
        if e['nodeid'] not in parents:raise ValueError('Unregistered raw test parent')
        if e['kind']=='parent':
            key=(e['nodeid'],e['phase'])
            if key in primary or e['subtest_index'] is not None or e['subtest_context'] is not None:raise ValueError('Repeated/mislabeled raw parent')
            primary[key]=e
        elif e['kind']=='subtest':
            ordinal=e['subtest_index'];next_ordinal=counters.get(e['nodeid'],0)+1
            if ordinal!=next_ordinal or e['phase']!='call' or not isinstance(e['subtest_context'],dict):raise ValueError('Missing/changed raw ordered subtest')
            counters[e['nodeid']]=ordinal;key=e['nodeid']+'::owned-subtest:'+str(ordinal)
            children[key]=e;contexts[key]=e['subtest_context']
        else:raise ValueError('Unknown raw event kind')
    case_ids=sorted(set(parents)|set(children))
    summary=dict(cases=case_ids,contexts=contexts,parents=parents,children=list(children),version=r['framework_version'])
    complete=(bool(parents) and set(starts)==set(parents)
        and set(primary)=={(name,phase) for name in parents for phase in ['setup','call','teardown']})
    if (not complete or r['collection_errors'] or r['session_exitstatus'] not in {0,1}
            or any(e['outcome']=='skipped' or e['expected_failure'] is not None or
                (e['phase']!='call' and e['outcome']!='passed') for e in events)):
        return 'incomplete_test_inventory_qualification',summary
    if reference is not None:
        if case_ids!=reference['cases'] or summary['version']!=reference['version']:return 'changed_case_population_qualification',summary
        if contexts!=reference['contexts']:return 'changed_subtest_context_qualification',summary
    failed=[e for e in events if e['phase']=='call' and e['outcome']=='failed']
    if not failed and r['session_exitstatus']==0:return 'all_finite_published_cases_pass',summary
    if reference is None:return 'original_reference_test_failure_qualification',summary
    if any(not e['assertion_failure'] or not e['exception_type'] for e in failed):return 'nonassertion_test_exception_qualification',summary
    if failed and r['session_exitstatus']==1:return 'completed_published_assertion_counterexample',summary
    return 'test_failure_without_retained_assertion_qualification',summary


def main():
    if (DATA/'admission.json').exists():raise ValueError('Existing completed v2 admission must be preserved')
    m=json.loads((DATA/'manifest.json').read_text());old=json.loads((PARENT/'admission.json').read_text())
    files={**m['source_files_sha256'],str(DATA/'manifest.json'):file_hash(DATA/'manifest.json'),
        str(REPORTS/'plan.json'):m['plan_sha256'],str(Path(__file__).resolve()):file_hash(__file__)}
    for name,digest in m['artifact_files_sha256'].items():files[str(DATA/name)]=digest
    for name,digest in m['repaired_execution_reports_sha256'].items():files[str(REPORTS/name)]=digest
    guard=FileBindings(files)
    def read(folder,name):return list(map(json.loads,(folder/name).read_text().splitlines()))
    old_positive=read(PARENT,'passing-originals.jsonl');old_pairs=read(PARENT,'pairs.jsonl')
    positives=read(DATA,'passing-originals.jsonl');pairs=read(DATA,'pairs.jsonl')
    positive_by={r['id']:r for r in positives};pair_by={r['id']:r for r in pairs}
    if len(positive_by)!=len(positives) or len(pair_by)!=len(pairs):raise ValueError('Distinct complete feedback views required')
    if any(positive_by.get(r['id'])!=r for r in old_positive) or any(pair_by.get(r['id'])!=r for r in old_pairs):
        raise ValueError('Every prior independently admitted view must remain exactly unchanged')
    rows=[r for r in read(SOURCE,'train.jsonl') if r['language']=='Python'];held={r['group_id'] for r in read(SOURCE,'valid.jsonl')}
    if len(rows)!=86 or any(r['split']!='train' or r['group_id'] in held for r in rows):raise ValueError('Original all86 source roles required')
    tokenizer=Tokenizer.from_file(str(TOKEN));encoder=Encoder(tokenizer,ModelConfig.load(ROOT/'configs/model.json'),SOURCE,448)
    suites=PublishedPythonSuitesV2(ROOT/'data/exercism-source-cache-v1/python')
    expected_positive={r['id'] for r in old_positive};expected_pairs={r['id'] for r in old_pairs}
    new_cases=0;new_parent_cases=0;subtests=0;qualifications=Counter()
    try:
        for row in rows:
            if row['exercise'] not in TARGETS:continue
            proof=json.loads((REPORTS/('module-'+row['exercise']+'.json')).read_text());spec=suites.suite(row)
            prompt=encoder.prompt(row);ids,labels=sequence(tokenizer,prompt,row['response'])
            digest=hashlib.sha256(row['response'].encode()).hexdigest()
            if (proof['id']!=row['id'] or proof['group_id']!=row['group_id'] or proof['split']!='train'
                    or proof['original_response']!=row['response'] or proof['prompt']!=prompt
                    or proof['test_sha256']!=spec['test_sha256'] or proof['source_response_sha256']!=digest
                    or proof['original_provided_helpers']!=spec.get('original_provided_helpers',{})):
                raise ValueError('Exact original repaired source, tests and helper bytes required')
            label,reference=raw_measure(proof['reference_execution'])
            if label!='all_finite_published_cases_pass' or proof['qualification']!=label:raise ValueError('Every repaired original must actually pass')
            positive=positive_by[row['id']+':passing-original']
            if (positive['source_id']!=row['id'] or positive['response']!=row['response'] or positive['prompt']!=prompt
                    or positive['context_tokens']!=len(ids) or positive['target_tokens']!=sum(y!=-100 for y in labels)
                    or positive['source_response_sha256']!=digest or positive['registered_cases']!=reference['cases']
                    or positive['reference_report_sha256']!=proof['reference_execution']['report_sha256']):raise ValueError('Whole original passing view differs')
            expected_positive.add(positive['id']);new_cases+=len(reference['cases']);new_parent_cases+=len(reference['parents']);subtests+=len(reference['children'])
            candidates=mutations(row['response'],row['id'])
            if len(candidates)!=len(proof['mutations']):raise ValueError('Every declared repair mutation must remain accounted for')
            for number,(candidate,entry) in enumerate(zip(candidates,proof['mutations']),1):
                if entry['response']!=candidate['response'] or entry['patch']!=candidate['patch'] or entry['number']!=number:
                    raise ValueError('Complete deterministic source patch differs')
                mid,mlabels=sequence(tokenizer,prompt,entry['response'])
                label,mutant=raw_measure(entry['execution'],reference);qualifications[label]+=1
                if label!=entry['qualification']:raise ValueError('Independent parent/subtest mutation classification differs')
                if label!='completed_published_assertion_counterexample':continue
                pair=pair_by[row['id']+':mutation:'+str(number)]
                failed=sorted(e['nodeid']+'::owned-subtest:'+str(e['subtest_index']) if e['kind']=='subtest' else e['nodeid']
                    for e in json.loads(next(line[len(PREFIX):] for line in entry['execution']['raw_inventory_output'].splitlines()
                        if line.startswith(PREFIX)))['events'] if e['phase']=='call' and e['outcome']=='failed')
                if (pair['source_id']!=row['id'] or pair['chosen']!=row['response'] or pair['rejected']!=entry['response']
                        or pair['prompt']!=prompt or pair['chosen_sha256']!=digest
                        or pair['rejected_sha256']!=hashlib.sha256(entry['response'].encode()).hexdigest()
                        or pair['chosen_context_tokens']!=len(ids) or pair['chosen_target_tokens']!=sum(y!=-100 for y in labels)
                        or pair['rejected_context_tokens']!=len(mid) or pair['rejected_target_tokens']!=sum(y!=-100 for y in mlabels)
                        or pair['registered_cases']!=reference['cases'] or pair['failed_cases']!=failed
                        or pair['reference_report_sha256']!=proof['reference_execution']['report_sha256']
                        or pair['mutant_report_sha256']!=entry['execution']['report_sha256']):raise ValueError('Whole exact source preference/mask/case evidence differs')
                expected_pairs.add(pair['id'])
        if expected_pairs!=set(pair_by) or expected_positive!=set(positive_by) or len(positives)!=86:
            raise ValueError('Every original and prior view must remain; no extras')
        for r in positives+pairs:
            if r['split']!='train' or r['human_preferences'] is not False or r['group_id'] in held:
                raise ValueError('All machine feedback stays in original training roles')
        cases=old['original_publisher_cases_passed']+new_cases
        if cases!=m['original_publisher_cases_passed'] or len(pairs)!=m['finite_assertion_pairs'] or dict(qualifications)!=m['new_mutation_qualifications']:
            raise ValueError('All aggregate native source counts must match')
        guard.check(full=True)
        result=dict(format='own-finite-published-Python-feedback-admission-v2',passed=True,
            completed_at=datetime.now(timezone.utc).isoformat(),candidate_manifest_sha256=file_hash(DATA/'manifest.json'),
            source_roles_only_train=True,machine_feedback_not_human_preferences=True,original_sources_accounted=86,
            complete_passing_originals=86,finite_assertion_pairs=len(pairs),original_publisher_cases_passed=cases,
            publisher_parent_test_cases=old['original_publisher_cases_passed']+new_parent_cases,publisher_subtest_cases=subtests,
            case_count_definition='Registered parent tests plus distinct completed subtests; not a count of independent assertions.',
            every_prior82positive_and153pair_view_identical=True,all_four_original_qualifications_resolved=True,
            every_repaired_source_helper_full_raw_events_and_BPE_mask_reconciled=True,
            changed_subtest_contexts_never_preference_labels=True,new_mutation_qualifications=dict(qualifications),
            held_group_overlap=0,encoder_truncated_sequences=0,original_full_code_and_replay_populations_unchanged=True,
            source_files_sha256=files,model_updates=0,goal_achieved=False)
        atomic_json(DATA/'admission.json',result)
        print(json.dumps({k:result[k] for k in ['passed','complete_passing_originals','finite_assertion_pairs',
            'publisher_parent_test_cases','publisher_subtest_cases','original_publisher_cases_passed','new_mutation_qualifications']},indent=2))
    finally:suites.close()


if __name__=='__main__':main()
