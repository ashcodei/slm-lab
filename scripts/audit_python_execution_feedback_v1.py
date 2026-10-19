"""Reconcile every complete Python publisher event, source, patch and own-BPE view."""
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
from slm_lab.python_execution_feedback_v1 import PublishedPythonSuites,extract_files,mutations
from slm_lab.model import ModelConfig
from slm_lab.training import Encoder

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/python-execution-feedback-v1'
REPORTS=ROOT/'reports/python-execution-feedback-v1'
SOURCE=ROOT/'data/exercism-support-context-v3'
TOKEN=ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json'
PREFIX='SLM_PYTHON_PUBLISHER_INVENTORY\t'


def independent_execution(record,reference=None):
    run=record['execution']
    if run['returncode']!=0 or run['limit'] is not None:return 'runtime_or_environment_qualification',None
    raw=record['raw_inventory_output']
    if hashlib.sha256(raw.encode()).hexdigest()!=record['report_sha256']:raise ValueError('Retained raw publisher bytes/hash mismatch')
    payload=[line[len(PREFIX):] for line in raw.splitlines() if line.startswith(PREFIX)]
    if len(payload)!=1:return 'incomplete_test_inventory_qualification',None
    value=json.loads(payload[0]);registered=value['registered'];started=value['started']
    if (value['format']!='own-complete-python-publisher-events-v1' or value['framework']!='pytest'
            or len(registered)!=len(set(registered)) or len(started)!=len(set(started))):raise ValueError('Original event schema/identity differs')
    event_list=value['events'];events={(e['nodeid'],e['phase']):e for e in event_list}
    # v1 cannot assign separate identities to pytest's dynamic unittest
    # subreports. They remain explicitly qualified and supply no labels.
    if len(events)!=len(event_list):return 'incomplete_test_inventory_qualification',value
    complete=(bool(registered) and set(started)==set(registered)
        and set(events)=={(case,phase) for case in registered for phase in ['setup','call','teardown']})
    if (not complete or value['collection_errors'] or value['session_exitstatus'] not in {0,1}
            or any(e['outcome']=='skipped' or e['expected_failure'] is not None or
                (e['phase']!='call' and e['outcome']!='passed') for e in event_list)):
        return 'incomplete_test_inventory_qualification',value
    if reference is not None and (sorted(registered)!=sorted(reference['registered'])
            or value['framework_version']!=reference['framework_version']):return 'changed_case_population_qualification',value
    failed=[e for e in event_list if e['phase']=='call' and e['outcome']=='failed']
    if not failed and value['session_exitstatus']==0:return 'all_finite_published_cases_pass',value
    if reference is None:return 'original_reference_test_failure_qualification',value
    if any(not e['assertion_failure'] or not e['exception_type'] for e in failed):return 'nonassertion_test_exception_qualification',value
    if failed and value['session_exitstatus']==1:return 'completed_published_assertion_counterexample',value
    return 'test_failure_without_retained_assertion_qualification',value


def main():
    if (DATA/'admission.json').exists():raise ValueError('Existing immutable admission must not be overwritten')
    manifest=json.loads((DATA/'manifest.json').read_text())
    plan=json.loads((REPORTS/'plan.json').read_text())
    files={**manifest['source_files_sha256'],str(DATA/'manifest.json'):file_hash(DATA/'manifest.json'),
        str(REPORTS/'plan.json'):manifest['plan_sha256'],str(Path(__file__).resolve()):file_hash(__file__)}
    for name,digest in manifest['execution_reports_sha256'].items():files[str(REPORTS/name)]=digest
    for name,digest in manifest['artifact_files_sha256'].items():files[str(DATA/name)]=digest
    guard=FileBindings(files)
    rows=[json.loads(line) for line in (SOURCE/'train.jsonl').read_text().splitlines()]
    held=[json.loads(line) for line in (SOURCE/'valid.jsonl').read_text().splitlines()]
    held_groups={r['group_id'] for r in held};rows=[r for r in rows if r['language']=='Python']
    if len(rows)!=86 or any(r['split']!='train' or r['group_id'] in held_groups for r in rows):raise ValueError('Original all86 train-only source roles required')
    pairs=list(map(json.loads,(DATA/'pairs.jsonl').read_text().splitlines()))
    positives=list(map(json.loads,(DATA/'passing-originals.jsonl').read_text().splitlines()))
    if len({r['id'] for r in pairs})!=len(pairs) or len({r['id'] for r in positives})!=len(positives):raise ValueError('Every admitted source view must be distinct')
    by_pair={r['id']:r for r in pairs};by_positive={r['source_id']:r for r in positives}
    tokenizer=Tokenizer.from_file(str(TOKEN));encoder=Encoder(tokenizer,ModelConfig.load(ROOT/'configs/model.json'),SOURCE,448)
    suites=PublishedPythonSuites(ROOT/'data/exercism-source-cache-v1/python')
    checked_pairs=set();checked_positives=set();qualification=Counter();mutation_qualification=Counter();cases=0
    try:
        for row in rows:
            report=json.loads((REPORTS/('module-'+row['exercise']+'.json')).read_text())
            prompt=encoder.prompt(row);response=row['response'];digest=hashlib.sha256(response.encode()).hexdigest()
            if (report['id']!=row['id'] or report['group_id']!=row['group_id'] or report['split']!='train'
                    or report['prompt']!=prompt or report['original_response']!=response or report['source_response_sha256']!=digest):
                raise ValueError('Every complete original role, prompt and source byte must match')
            try:spec=suites.suite(row)
            except (ValueError,KeyError,StopIteration):
                if report['qualification']!='original_layout_review' or report['mutations']:raise ValueError('Original source layout qualification changed')
                qualification[report['qualification']]+=1;continue
            try:ids,labels=sequence(tokenizer,prompt,response)
            except ValueError:
                if report['qualification']!='complete_original_context_exceeds4096':raise ValueError('Whole context qualification changed')
                qualification[report['qualification']]+=1;continue
            label,reference=independent_execution(report['reference_execution'])
            if label!=report['qualification'] or report['test_sha256']!=spec['test_sha256']:
                raise ValueError('Original full publisher execution classification differs')
            qualification[label]+=1
            if label!='all_finite_published_cases_pass':
                if report['mutations']:raise ValueError('Nonpassing references cannot create preferences')
                continue
            positive=by_positive.get(row['id'])
            if (positive is None or positive['response']!=response or positive['prompt']!=prompt or positive['split']!='train'
                    or positive['group_id']!=row['group_id'] or positive['human_preferences'] is not False
                    or positive['source_response_sha256']!=digest or positive['context_tokens']!=len(ids)
                    or positive['target_tokens']!=sum(y!=-100 for y in labels)
                    or positive['registered_cases']!=sorted(reference['registered'])
                    or positive['reference_report_sha256']!=report['reference_execution']['report_sha256']):
                raise ValueError('Complete original passing program/BPE/cases differ')
            checked_positives.add(row['id']);cases+=len(reference['registered'])
            expected=mutations(response,row['id'])
            if len(expected)!=len(report['mutations']):raise ValueError('Every declared bounded mutation must remain accounted for')
            for number,(candidate,entry) in enumerate(zip(expected,report['mutations']),1):
                patch=candidate['patch']
                if (entry['number']!=number or entry['patch']!=patch or entry['response']!=candidate['response']
                        or response[patch['start']:patch['end']]!=patch['old']
                        or candidate['response']!=response[:patch['start']]+patch['new']+response[patch['end']:]):
                    raise ValueError('Every source token patch must reconstruct the complete response exactly')
                extract_files(entry['response'],spec['expected_solution_files'])
                try:mid,mlabels=sequence(tokenizer,prompt,entry['response'])
                except ValueError:
                    if entry['qualification']!='complete_mutated_context_exceeds4096':raise ValueError('Complete mutation context qualification differs')
                    mutation_qualification[entry['qualification']]+=1;continue
                label,raw=independent_execution(entry['execution'],reference)
                if label!=entry['qualification']:raise ValueError('Independent complete raw mutant classification differs')
                mutation_qualification[label]+=1
                if label!='completed_published_assertion_counterexample':continue
                key=row['id']+':mutation:'+str(number);pair=by_pair.get(key)
                failed=sorted(e['nodeid'] for e in raw['events'] if e['phase']=='call' and e['outcome']=='failed')
                if (pair is None or pair['source_id']!=row['id'] or pair['group_id']!=row['group_id'] or pair['split']!='train'
                        or pair['human_preferences'] is not False or pair['prompt']!=prompt or pair['chosen']!=response
                        or pair['rejected']!=entry['response'] or pair['chosen_sha256']!=digest
                        or pair['rejected_sha256']!=hashlib.sha256(entry['response'].encode()).hexdigest()
                        or pair['mutation']!=patch or pair['chosen_context_tokens']!=len(ids)
                        or pair['chosen_target_tokens']!=sum(y!=-100 for y in labels) or pair['rejected_context_tokens']!=len(mid)
                        or pair['rejected_target_tokens']!=sum(y!=-100 for y in mlabels)
                        or pair['registered_cases']!=sorted(reference['registered']) or pair['failed_cases']!=failed
                        or pair['original_test_sha256']!=spec['test_sha256']
                        or pair['reference_report_sha256']!=report['reference_execution']['report_sha256']
                        or pair['mutant_report_sha256']!=entry['execution']['report_sha256']):raise ValueError('Every complete original train-only preference/body/mask/case/exception must match')
                checked_pairs.add(key)
        if checked_pairs!=set(by_pair) or checked_positives!=set(by_positive):raise ValueError('No extra or missing training views allowed')
        if (dict(qualification)!=manifest['original_source_qualifications'] or dict(mutation_qualification)!=manifest['mutation_qualifications']
                or len(pairs)!=manifest['finite_assertion_pairs'] or len(positives)!=manifest['complete_passing_originals']):raise ValueError('Full accounting/counts differ')
        guard.check(full=True)
        result=dict(format='own-finite-published-Python-feedback-admission-v1',completed_at=datetime.now(timezone.utc).isoformat(),
            passed=True,candidate_manifest_sha256=file_hash(DATA/'manifest.json'),source_roles_only_train=True,
            machine_feedback_not_human_preferences=True,original_sources_accounted=86,complete_passing_originals=len(positives),
            original_publisher_cases_passed=cases,finite_assertion_pairs=len(pairs),source_groups_with_pairs=manifest['source_groups_with_pairs'],
            original_source_qualifications=dict(qualification),mutation_qualifications=dict(mutation_qualification),
            every_original_publisher_test_source_and_raw_phase_inventory_reconciled=True,
            full_original_and_mutant_prompts_programs_EOS_masks_preserved=True,encoder_truncated_sequences=0,
            original_full_code_and_replay_populations_unchanged=True,held_group_overlap=0,
            source_files_sha256=files,model_updates=0,goal_achieved=False,
            scope='Supplemental finite Python publisher assertion feedback only; no high-end model capability, human preferences or production readiness claim.')
        atomic_json(DATA/'admission.json',result)
        print(json.dumps({k:result[k] for k in ['passed','original_sources_accounted','complete_passing_originals',
            'original_publisher_cases_passed','finite_assertion_pairs','original_source_qualifications','mutation_qualifications']},indent=2))
    finally:suites.close()


if __name__=='__main__':main()
