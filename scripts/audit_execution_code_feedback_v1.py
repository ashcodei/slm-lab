"""Reconcile every original source, mutation, raw JVM case and training pair."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.execution_code_feedback_v1 import mutations
from slm_lab.exercise_java_runner import PublishedSuites,extract_files
from slm_lab.exercise_java_evaluation import question
from slm_lab.model import ModelConfig
from slm_lab.training import Encoder

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/execution-code-feedback-v1'
REPORTS=ROOT/'reports/execution-code-feedback-v1'
SOURCE=ROOT/'data/exercism-support-context-v3'


def verdict(execution,expected=None):
    """Read actual raw event arrays rather than trusting parsed score flags."""
    compile_pass=execution.get('returncode')==0 and execution.get('limit') is None
    if compile_pass!=execution.get('compilation_passed'):raise ValueError('Compiler flag differs from actual process outcome')
    if not compile_pass:return 'compiler_or_environment_qualification'
    runtime=execution.get('execution',{})
    if runtime.get('returncode')!=0 or runtime.get('limit') is not None:return 'runtime_or_environment_qualification'
    raw=execution.get('raw_inventory_output')
    if not isinstance(raw,str):return 'incomplete_test_inventory_qualification'
    entries=[line[len('SLM_JAVA_INVENTORY\t'):] for line in raw.splitlines() if line.startswith('SLM_JAVA_INVENTORY\t')]
    if len(entries)!=1:raise ValueError('Exactly one complete actual raw JVM inventory required')
    original=json.loads(entries[0]);sets={}
    for key in ['registered','started','passed','failed','skipped','container_errors']:
        values=original[key]
        if len(values)!=len(set(values)):raise ValueError('Duplicate actual Java case event')
        sets[key]=set(values)
    if original['junit4_version']!='4.13' or original['mode'] not in {'junit4','jupiter'}:
        raise ValueError('Original installed test framework differs')
    parsed=dict(cases=sorted(sets['registered']),started_cases=sorted(sets['started']),passed_cases=sorted(sets['passed']),
        failed=original['failed'],skipped=original['skipped'],container_errors=original['container_errors'],
        framework=original['mode'],junit4_version='4.13',all_passed=(bool(sets['registered'])
            and sets['registered']==sets['started']==sets['passed'] and not sets['failed'] and not sets['skipped'] and not sets['container_errors']))
    if parsed!=execution['inventory'] or hashlib.sha256(raw.encode()).hexdigest()!=execution['report_sha256']:
        raise ValueError('All actual raw event arrays must match recorded evidence')
    if (not sets['registered'] or sets['skipped'] or sets['container_errors'] or sets['started']!=sets['registered']
            or sets['passed']&sets['failed'] or sets['passed']|sets['failed']!=sets['registered']):
        return 'incomplete_test_inventory_qualification'
    if expected is not None and parsed['cases']!=expected:return 'changed_case_population_qualification'
    if parsed['all_passed']:
        if not execution['published_tests_passed']:raise ValueError('Passing publisher flag differs')
        return 'all_finite_published_cases_pass'
    if expected is None:return 'original_reference_test_failure_qualification'
    prefixes=['TEST_FAILURE: java.lang.AssertionError','TEST_FAILURE: org.opentest4j.AssertionFailedError',
        'TEST_FAILURE: org.junit.ComparisonFailure','TEST_FAILURE: junit.framework.AssertionFailedError',
        'TEST_FAILURE: junit.framework.ComparisonFailure']
    assertion=any(any(line.startswith(prefix+':') or line==prefix for prefix in prefixes)
        for line in runtime.get('diagnostic','').splitlines())
    if sets['failed'] and assertion:
        if execution['published_tests_passed']:raise ValueError('Failed publisher flag differs')
        return 'completed_published_assertion_counterexample'
    return 'test_failure_without_retained_assertion_qualification'


def main():
    if (DATA/'admission.json').exists():raise ValueError('Preserve prior complete feedback admission')
    manifest=json.loads((DATA/'manifest.json').read_text());plan=json.loads((REPORTS/'plan.json').read_text())
    if manifest['plan_sha256']!=file_hash(REPORTS/'plan.json') or manifest['training_admitted'] is not False:
        raise ValueError('Exact unadmitted completed candidate source required')
    files={**manifest['source_files_sha256'],str(DATA/'manifest.json'):file_hash(DATA/'manifest.json'),
        str(REPORTS/'plan.json'):file_hash(REPORTS/'plan.json'),str(Path(__file__).resolve()):file_hash(__file__),
        **{str(DATA/name):digest for name,digest in manifest['artifact_files_sha256'].items()},
        **{str(REPORTS/name):digest for name,digest in manifest['execution_reports_sha256'].items()}}
    guard=FileBindings(files)
    all_train=list(map(json.loads,(SOURCE/'train.jsonl').open()));held=list(map(json.loads,(SOURCE/'valid.jsonl').open()))
    rows=[r for r in all_train if r['language']=='Java'];pair_rows=list(map(json.loads,(DATA/'pairs.jsonl').open()))
    if len(rows)!=109 or {r['group_id'] for r in rows}&{r['group_id'] for r in held}:
        raise ValueError('All original109train modules and zero held group overlap required')
    if plan['source_population']!=[{k:r[k] for k in ['id','group_id','language','split']} for r in rows]:
        raise ValueError('Every original sealed source identity and order must match')
    if len({r['id'] for r in pair_rows})!=len(pair_rows):raise ValueError('Unique complete machine-feedback pairs required')
    indexed={r['id']:r for r in pair_rows};expected_pair_ids=[];positives=[];cases=0
    verdicts=Counter();mutant_verdicts=Counter()
    suites=PublishedSuites(ROOT/'data/exercism-source-cache-v1/java')
    token_path=ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json';tok=Tokenizer.from_file(str(token_path))
    encoder=Encoder(tok,ModelConfig.load(ROOT/'configs/model.json'),SOURCE,448)
    try:
        for row in rows:
            record=json.loads((REPORTS/('module-'+row['exercise']+'.json')).read_text())
            if (record['id']!=row['id'] or record['split']!='train' or record['group_id']!=row['group_id']
                    or record['source_response_sha256']!=hashlib.sha256(row['response'].encode()).hexdigest()):
                raise ValueError('Exact original training program/identity required')
            try:suite=suites.suite(row);suites.reference(row)
            except (ValueError,KeyError,StopIteration):
                if record['qualification']!='original_layout_or_environment_review':raise
                verdicts[record['qualification']]+=1;continue
            prompt=encoder.prompt(question(row,suite));prefix=[1]+tok.encode(prompt).ids
            body=tok.encode(row['response']).ids+[2];context=len(prefix)+len(body)
            if context>4096:
                if record['qualification']!='whole_original_extended_context_exceeds4096':raise ValueError('Unfit original must remain qualified')
                verdicts[record['qualification']]+=1;continue
            if (record['prompt']!=prompt or record['original_response']!=row['response'] or record['original_context_tokens']!=context
                    or record['original_target_tokens']!=len(body) or record['test_sha256']!=suite['test_sha256']
                    or record['editor_sha256']!=suite['editor_sha256']):
                raise ValueError('Every original whole question/program and publisher test/support byte must match')
            label=verdict(record['reference_execution']);verdicts[label]+=1
            if label!=record['qualification']:raise ValueError('Original reference qualification differs from actual outcomes')
            if label!='all_finite_published_cases_pass':
                if record['mutations']:raise ValueError('Unproven original reference cannot label mutants')
                continue
            expected=record['reference_execution']['inventory']['cases'];cases+=len(expected)
            positives.append(dict(id=row['id'],group_id=row['group_id'],split='train',language='Java',
                prompt=prompt,response=row['response'],source_response_sha256=record['source_response_sha256'],
                context_tokens=context,target_tokens=len(body),source=row['source'],revision=row['revision'],
                license=row['license'],license_url=row['license_url'],authors=row['authors'],
                feedback_source='machine_execution_of_original_publisher_tests',human_preferences=False,
                original_reference_report_sha256=record['reference_execution']['report_sha256']))
            generated=mutations(row['response'],row['id'])
            if len(generated)!=len(record['mutations']):raise ValueError('All deterministic attempted mutations must remain accounted for')
            for number,(expected_mutant,observed) in enumerate(zip(generated,record['mutations']),1):
                if any(observed[k]!=expected_mutant[k] for k in ['patch','response','response_sha256']):
                    raise ValueError('Mutation changed more than its declared original code token')
                extract_files(observed['response'],suite['expected_solution_files'])
                mutant_body=tok.encode(observed['response']).ids+[2]
                if len(prefix)+len(mutant_body)>4096:
                    if observed['qualification']!='complete_mutated_response_exceeds4096':raise ValueError('Mutant context qualification missing')
                    mutant_verdicts[observed['qualification']]+=1;continue
                measured=verdict(observed['execution'],expected);mutant_verdicts[measured]+=1
                if measured!=observed['qualification']:raise ValueError('Mutation label differs from raw original publisher case events')
                if measured!='completed_published_assertion_counterexample':continue
                ident=row['id']+':mutation:'+str(number);expected_pair_ids.append(ident);pair=indexed[ident]
                checks=dict(source_id=row['id'],group_id=row['group_id'],split='train',language='Java',prompt=prompt,
                    chosen=row['response'],rejected=observed['response'],chosen_sha256=record['source_response_sha256'],
                    rejected_sha256=observed['response_sha256'],mutation=observed['patch'],mutation_number=number,
                    chosen_context_tokens=context,chosen_target_tokens=len(body),rejected_context_tokens=len(prefix)+len(mutant_body),
                    rejected_target_tokens=len(mutant_body),original_test_sha256=suite['test_sha256'],
                    original_editor_sha256=suite['editor_sha256'],registered_cases=expected,
                    failed_cases=observed['execution']['inventory']['failed'],reference_report_sha256=record['reference_execution']['report_sha256'],
                    mutant_report_sha256=observed['execution']['report_sha256'],
                    feedback_source='machine_execution_of_original_publisher_tests',human_preferences=False)
                if any(pair[k]!=v for k,v in checks.items()):raise ValueError('Every complete labeled pair must reconstruct from actual execution and original source')
        if expected_pair_ids!=[r['id'] for r in pair_rows] or len(pair_rows)!=manifest['pairs']:
            raise ValueError('Exact complete source-ordered finite assertion pair population required')
        if dict(verdicts)!=manifest['original_module_qualifications'] or dict(mutant_verdicts)!=manifest['mutation_qualifications']:
            raise ValueError('All source and mutant qualification counts must reconstruct')
        if len(verdicts)==0 or sum(verdicts.values())!=109 or not pair_rows:
            raise ValueError('All109original sources and nonempty proven supplemental pairs required')
        guard.check(full=True);suites.guard.check(full=True)
        with (DATA/'passing-originals.jsonl').open('x') as out:
            for row in positives:out.write(json.dumps(row,separators=(',',':'))+'\n')
        source_files={**files,str(DATA/'passing-originals.jsonl'):file_hash(DATA/'passing-originals.jsonl')}
        atomic_json(DATA/'admission.json',dict(format='own-finite-published-execution-feedback-admission-v1',passed=True,
            completed_at=datetime.now(timezone.utc).isoformat(),candidate_manifest_sha256=file_hash(DATA/'manifest.json'),
            source_files_sha256=source_files,original_sources_accounted=109,source_roles_only_train=True,
            complete_passing_originals=len(positives),complete_original_registered_cases=cases,
            finite_assertion_pairs=len(pair_rows),source_groups_with_pairs=len({p['group_id'] for p in pair_rows}),
            every_raw_case_inventory_and_mutation_and_full_pair_reconstructed=True,all_masks_and_EOS_fit_without_clipping=True,
            all_original_test_and_support_bytes_match=True,held_role_or_case_labels_used_for_optimization=False,
            ordinary_full_code_and_replay_source_populations_unchanged=True,machine_feedback_not_human_preferences=True,
            source_qualifications=dict(verdicts),mutation_qualifications=dict(mutant_verdicts),
            admits='Supplemental train-only finite-execution preference pairs and complete passing-original CE views only. Does not replace original full-code sources or certify programs on all inputs.',
            final_production_source_contract_integrated=False,model_updates=0,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(original_sources_accounted=109,passing_originals=len(positives),
            original_published_cases=cases,verified_assertion_pairs=len(pair_rows),groups_with_pairs=len({p['group_id'] for p in pair_rows})),indent=2))
    finally:suites.close()


if __name__=='__main__':main()
