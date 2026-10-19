"""Retain all v1 feedback; repair only the four qualified original Python suites."""
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
from slm_lab.python_execution_feedback_v1 import extract_files,mutations
from slm_lab.python_execution_feedback_v2 import PublishedPythonSuitesV2,classify
from slm_lab.training import Encoder

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'data/python-execution-feedback-v1'
DATA=ROOT/'data/python-execution-feedback-v2'
REPORTS=ROOT/'reports/python-execution-feedback-v2'
SOURCE=ROOT/'data/exercism-support-context-v3'
TOKEN=ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json'
TARGETS={'electric-bill','meltdown-mitigation','diffie-hellman','error-handling'}


def main():
    if DATA.exists() or REPORTS.exists():raise ValueError('New distinct owned v2 outputs required')
    log=ROOT/'reports/python-execution-feedback-v2-focused-tests.log'
    if '4 passed' not in log.read_text():raise ValueError('Native subtest and original-four checks required')
    admission=json.loads((PARENT/'admission.json').read_text())
    parent=json.loads((PARENT/'manifest.json').read_text())
    if not admission['passed'] or admission['complete_passing_originals']!=82 or admission['finite_assertion_pairs']!=153:
        raise ValueError('Exact complete independently admitted v1 parent required')
    rows=[r for r in map(json.loads,(SOURCE/'train.jsonl').read_text().splitlines()) if r['language']=='Python']
    if len(rows)!=86 or sum(r['exercise'] in TARGETS for r in rows)!=4:raise ValueError('All original86 Python sources and exact four repairs required')
    held={r['group_id'] for r in map(json.loads,(SOURCE/'valid.jsonl').read_text().splitlines())}
    if any(r['split']!='train' or r['group_id'] in held for r in rows):raise ValueError('Original train roles must remain disjoint from held groups')
    suites=PublishedPythonSuitesV2(ROOT/'data/exercism-source-cache-v1/python')
    paths=[PARENT/'manifest.json',PARENT/'admission.json',PARENT/'pairs.jsonl',PARENT/'passing-originals.jsonl',
        SOURCE/'manifest.json',SOURCE/'train.jsonl',SOURCE/'valid.jsonl',TOKEN,Path(__file__).resolve(),log]
    files={**admission['source_files_sha256'],**{str(p):h for p,h in suites.guard.files.items()},**{str(p):file_hash(p) for p in paths}}
    guard=FileBindings(files);tokenizer=Tokenizer.from_file(str(TOKEN));encoder=Encoder(tokenizer,ModelConfig.load(ROOT/'configs/model.json'),SOURCE,448)
    DATA.mkdir();REPORTS.mkdir()
    atomic_json(REPORTS/'plan.json',dict(format='own-complete-Python-execution-feedback-plan-v2',
        created_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,source_population=[r['id'] for r in rows],
        repaired_original_ids=[r['id'] for r in rows if r['exercise'] in TARGETS],
        all82previous_positives_and153previous_pairs_retained_exactly=True,
        original_parent_and_subtest_contexts_preserved=True,changed_subtest_context_not_a_preference=True,
        provided_helper_from_original_archive_only=True,training_admitted=False,model_updates=0,goal_achieved=False))
    positives=list(map(json.loads,(PARENT/'passing-originals.jsonl').read_text().splitlines()))
    pairs=list(map(json.loads,(PARENT/'pairs.jsonl').read_text().splitlines()))
    proofs={};new_qualifications=Counter()
    try:
        for row in rows:
            if row['exercise'] not in TARGETS:continue
            guard.check();spec=suites.suite(row);prompt=encoder.prompt(row);ids,labels=sequence(tokenizer,prompt,row['response'])
            original=suites.run(extract_files(row['response'],spec['expected_solution_files']),spec)
            qualified=classify(original)
            if qualified!='all_finite_published_cases_pass':raise ValueError('Repaired original full suite did not pass: '+row['id']+' '+qualified)
            sha=hashlib.sha256(row['response'].encode()).hexdigest()
            proof=dict(id=row['id'],group_id=row['group_id'],split='train',language='Python',backend_revision=2,
                prompt=prompt,original_response=row['response'],source_response_sha256=sha,
                original_context_tokens=len(ids),original_target_tokens=sum(y!=-100 for y in labels),
                expected_solution_files=spec['expected_solution_files'],test_sha256=spec['test_sha256'],
                original_provided_helpers=spec.get('original_provided_helpers',{}),reference_execution=original,
                qualification=qualified,mutations=[])
            positives.append(dict(id=row['id']+':passing-original',source_id=row['id'],group_id=row['group_id'],split='train',
                language='Python',prompt=prompt,response=row['response'],source_response_sha256=sha,context_tokens=len(ids),
                target_tokens=proof['original_target_tokens'],registered_cases=original['inventory']['cases'],
                original_test_sha256=spec['test_sha256'],reference_report_sha256=original['report_sha256'],human_preferences=False,
                feedback_source='machine_execution_of_original_publisher_tests',backend_revision=2))
            for number,mutant in enumerate(mutations(row['response'],row['id']),1):
                mid,mlabels=sequence(tokenizer,prompt,mutant['response'])
                measured=suites.run(extract_files(mutant['response'],spec['expected_solution_files']),spec)
                label=classify(measured,original['inventory']);new_qualifications[label]+=1
                entry=dict(number=number,**mutant,execution=measured,qualification=label,
                    context_tokens=len(mid),target_tokens=sum(y!=-100 for y in mlabels));proof['mutations'].append(entry)
                if label!='completed_published_assertion_counterexample':continue
                pairs.append(dict(id=row['id']+':mutation:'+str(number),source_id=row['id'],group_id=row['group_id'],language='Python',
                    split='train',prompt=prompt,chosen=row['response'],rejected=mutant['response'],chosen_sha256=sha,
                    rejected_sha256=mutant['response_sha256'],mutation=mutant['patch'],mutation_number=number,
                    chosen_context_tokens=len(ids),chosen_target_tokens=proof['original_target_tokens'],rejected_context_tokens=len(mid),
                    rejected_target_tokens=entry['target_tokens'],original_test_sha256=spec['test_sha256'],
                    registered_cases=original['inventory']['cases'],failed_cases=measured['inventory']['assertion_failed_cases'],
                    reference_report_sha256=original['report_sha256'],mutant_report_sha256=measured['report_sha256'],
                    feedback_source='machine_execution_of_original_publisher_tests',human_preferences=False,backend_revision=2,
                    label_scope='Complete original publisher parent/subtests pass; one-token mutant completes identical case and subtest-context inventories with assertion-only failures. Finite tests.'))
            proofs[row['id']]=proof;atomic_json(REPORTS/('module-'+row['exercise']+'.json'),proof)
            print('repaired',row['id'],'cases',len(original['inventory']['cases']),'total_pairs',len(pairs),flush=True)
        guard.check(full=True)
        for name,values in [('pairs.jsonl',pairs),('passing-originals.jsonl',positives)]:
            (DATA/name).write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in values))
        cases=admission['original_publisher_cases_passed']+sum(len(p['reference_execution']['inventory']['cases']) for p in proofs.values())
        result=dict(format='own-published-Python-execution-feedback-candidate-v2',completed_at=datetime.now(timezone.utc).isoformat(),
            parent_candidate_manifest_sha256=file_hash(PARENT/'manifest.json'),parent_admission_sha256=file_hash(PARENT/'admission.json'),
            original_source_manifest_sha256=file_hash(SOURCE/'manifest.json'),tokenizer_sha256=file_hash(TOKEN),original_training_sources=86,
            complete_passing_originals=len(positives),original_publisher_cases_passed=cases,finite_assertion_pairs=len(pairs),
            retained_parent_positives=82,retained_parent_pairs=153,repaired_original_sources=4,
            source_groups_with_pairs=len({p['group_id'] for p in pairs}),new_mutation_qualifications=dict(new_qualifications),
            plan_sha256=file_hash(REPORTS/'plan.json'),source_files_sha256=files,
            artifact_files_sha256={p.name:file_hash(p) for p in DATA.glob('*.jsonl')},
            repaired_execution_reports_sha256={p.name:file_hash(p) for p in REPORTS.glob('module-*.json')},
            training_admitted=False,human_preferences=False,model_updates=0,goal_achieved=False,
            scope='All86 original Python positives now have complete native publisher evidence. All prior admitted views retained. Counts include distinct parent/subtest cases. No full-model update, human feedback or capability claim.')
        atomic_json(DATA/'manifest.json',result);print(json.dumps({k:result[k] for k in ['complete_passing_originals',
            'original_publisher_cases_passed','finite_assertion_pairs','new_mutation_qualifications']},indent=2))
    finally:suites.close()


if __name__=='__main__':main()
