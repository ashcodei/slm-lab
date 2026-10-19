"""Run all86 original Python training publisher suites and bounded code mutations."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.execution_code_feedback_v1 import sequence
from slm_lab.model import ModelConfig
from slm_lab.python_execution_feedback_v1 import PublishedPythonSuites,classify,extract_files,mutations
from slm_lab.training import Encoder

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/exercism-support-context-v3'
DATA=ROOT/'data/python-execution-feedback-v1'
REPORTS=ROOT/'reports/python-execution-feedback-v1'
TOKEN=ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json'
MAX_REPORT=512*1024**2
MAX_DATA=32*1024**2


def main():
    if DATA.exists() or REPORTS.exists():raise ValueError('New owned execution directories required; never implicitly restart')
    if '8 passed' not in (ROOT/'reports/python-execution-feedback-v1-final-tests.log').read_text():
        raise ValueError('Actual confined native/collection/mutation/source-role checks required')
    if shutil.disk_usage(ROOT).free<26*1024**3+MAX_REPORT+MAX_DATA:raise ValueError('Original reserve and full report/data commitments required')
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    for name in ['train.jsonl','valid.jsonl']:
        if file_hash(SOURCE/name)!=manifest['files_sha256'][name]:raise ValueError('Exact original sealed source data required')
    train=list(map(json.loads,(SOURCE/'train.jsonl').read_text().splitlines()))
    held=list(map(json.loads,(SOURCE/'valid.jsonl').read_text().splitlines()))
    if {r['group_id'] for r in train}&{r['group_id'] for r in held}:raise ValueError('Original held group cannot enter any execution label')
    rows=[r for r in train if r['language']=='Python']
    if len(rows)!=86 or len({r['id'] for r in rows})!=86 or any(r['split']!='train' or r['license']!='MIT' for r in rows):
        raise ValueError('All86 original Python training sources must remain present')
    suites=PublishedPythonSuites(ROOT/'data/exercism-source-cache-v1/python')
    paths=[Path(__file__).resolve(),SOURCE/'manifest.json',SOURCE/'train.jsonl',SOURCE/'valid.jsonl',TOKEN,
        ROOT/'configs/model.json',ROOT/'src/slm_lab/execution_code_feedback_v1.py',ROOT/'src/slm_lab/training.py',
        ROOT/'src/slm_lab/model.py',ROOT/'reports/python-execution-feedback-v1-final-tests.log']
    files={**{str(p):d for p,d in suites.guard.files.items()},**{str(p):file_hash(p) for p in paths}}
    guard=FileBindings(files)
    # Integrity mismatches abort before any qualification or negative label.
    for row in rows:
        for item in row['source_files']+row['published_test_artifacts']:
            if hashlib.sha256(suites.read_item(item)).hexdigest()!=item['sha256']:
                raise ValueError('Original publisher source/test integrity mismatch')
    tokenizer=Tokenizer.from_file(str(TOKEN));encoder=Encoder(tokenizer,ModelConfig.load(ROOT/'configs/model.json'),SOURCE,448)
    if file_hash(TOKEN)!=manifest['tokenizer_sha256']:raise ValueError('Own original tokenizer required')
    DATA.mkdir();REPORTS.mkdir()
    atomic_json(REPORTS/'plan.json',dict(format='own-complete-Python-execution-feedback-plan-v1',
        created_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,runtime=suites.runtime,
        sources=[dict(id=r['id'],group_id=r['group_id'],split=r['split'],language=r['language']) for r in rows],
        original_sources=86,maximum_CPU_workers=4,mutations_per_passing_source=4,seed=930031,
        complete_original_prompts_and_programs=True,all_declared_publisher_test_files_unchanged=True,
        maximum_report_bytes=MAX_REPORT,maximum_data_bytes=MAX_DATA,minimum_free_bytes=26*1024**3,
        original_code_and_replay_populations_unchanged=True,training_admitted=False,model_updates=0,
        human_preferences=False,goal_achieved=False))
    def boundary(full=False):
        guard.check(full=full)
        if shutil.disk_usage(ROOT).free<26*1024**3:raise ValueError('Original persistent reserve must remain funded')
        if sum(p.stat().st_size for p in REPORTS.glob('*.json'))>MAX_REPORT:raise ValueError('Complete execution report budget exceeded')
    def run(row):
        boundary();prompt=encoder.prompt(row)
        record=dict(id=row['id'],group_id=row['group_id'],split='train',language='Python',prompt=prompt,
            original_response=row['response'],source_response_sha256=hashlib.sha256(row['response'].encode()).hexdigest(),mutations=[])
        try:spec=suites.suite(row)
        except (ValueError,KeyError,StopIteration) as error:
            return dict(**record,qualification='original_layout_review',detail=str(error)),[],None
        try:ids,labels=sequence(tokenizer,prompt,row['response'])
        except ValueError:return dict(**record,qualification='complete_original_context_exceeds4096'),[],None
        record.update(original_context_tokens=len(ids),original_target_tokens=sum(y!=-100 for y in labels),
            expected_solution_files=spec['expected_solution_files'],test_sha256=spec['test_sha256'])
        observed=suites.run(extract_files(row['response'],spec['expected_solution_files']),spec)
        label=classify(observed);record.update(reference_execution=observed,qualification=label)
        if label!='all_finite_published_cases_pass':return record,[],None
        positive=dict(id=row['id']+':passing-original',source_id=row['id'],group_id=row['group_id'],split='train',
            language='Python',prompt=prompt,response=row['response'],source_response_sha256=record['source_response_sha256'],
            context_tokens=len(ids),target_tokens=record['original_target_tokens'],registered_cases=observed['inventory']['cases'],
            original_test_sha256=spec['test_sha256'],reference_report_sha256=observed['report_sha256'],human_preferences=False,
            feedback_source='machine_execution_of_original_publisher_tests')
        pairs=[]
        for number,mutant in enumerate(mutations(row['response'],row['id']),1):
            entry=dict(number=number,**mutant)
            try:mid,mlabels=sequence(tokenizer,prompt,mutant['response'])
            except ValueError:
                record['mutations'].append(dict(**entry,qualification='complete_mutated_context_exceeds4096'));continue
            measured=suites.run(extract_files(mutant['response'],spec['expected_solution_files']),spec)
            qualified=classify(measured,observed['inventory'])
            entry.update(execution=measured,qualification=qualified,context_tokens=len(mid),target_tokens=sum(y!=-100 for y in mlabels))
            record['mutations'].append(entry)
            if qualified=='completed_published_assertion_counterexample':
                pairs.append(dict(id=row['id']+':mutation:'+str(number),source_id=row['id'],group_id=row['group_id'],
                    language='Python',split='train',prompt=prompt,chosen=row['response'],rejected=mutant['response'],
                    chosen_sha256=record['source_response_sha256'],rejected_sha256=mutant['response_sha256'],
                    mutation=mutant['patch'],mutation_number=number,chosen_context_tokens=len(ids),
                    chosen_target_tokens=record['original_target_tokens'],rejected_context_tokens=len(mid),
                    rejected_target_tokens=entry['target_tokens'],original_test_sha256=spec['test_sha256'],
                    registered_cases=observed['inventory']['cases'],failed_cases=measured['inventory']['assertion_failed_cases'],
                    reference_report_sha256=observed['report_sha256'],mutant_report_sha256=measured['report_sha256'],
                    feedback_source='machine_execution_of_original_publisher_tests',human_preferences=False,
                    label_scope='Original full program passes every publisher case; a one-token mutant completes the identical full case inventory with assertion failures and no other test exceptions. Finite tests only.'))
        return record,pairs,positive
    completed={};pair_rows={};positives={}
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs={pool.submit(run,row):row for row in rows}
            for job in as_completed(jobs):
                row=jobs[job];record,pairs,positive=job.result();completed[row['id']]=record;pair_rows[row['id']]=pairs
                if positive is not None:positives[row['id']]=positive
                atomic_json(REPORTS/('module-'+row['exercise']+'.json'),record)
                atomic_json(REPORTS/'status.json',dict(status='executing all86 original Python training modules',
                    updated_at=datetime.now(timezone.utc).isoformat(),completed_sources=len(completed),expected_sources=86,
                    complete_passing_originals=len(positives),finite_assertion_pairs=sum(map(len,pair_rows.values())),
                    training_admitted=False,model_updates=0,goal_achieved=False))
                print('original_sources',len(completed),'of86;passing',len(positives),'assertion_pairs',sum(map(len,pair_rows.values())),flush=True)
        boundary(True)
        if set(completed)!={r['id'] for r in rows}:raise ValueError('Every original training source must remain accounted for')
        for filename,values in [('pairs.jsonl',[p for row in rows for p in pair_rows[row['id']]]),
                ('passing-originals.jsonl',[positives[row['id']] for row in rows if row['id'] in positives])]:
            (DATA/filename).write_text(''.join(json.dumps(v,separators=(',',':'))+'\n' for v in values))
        if sum(p.stat().st_size for p in DATA.glob('*.jsonl'))>MAX_DATA:raise ValueError('Full complete feedback data budget exceeded')
        final=dict(format='own-published-Python-execution-feedback-candidate-v1',completed_at=datetime.now(timezone.utc).isoformat(),
            original_source_manifest_sha256=file_hash(SOURCE/'manifest.json'),tokenizer_sha256=file_hash(TOKEN),
            original_training_sources=86,complete_passing_originals=len(positives),finite_assertion_pairs=sum(map(len,pair_rows.values())),
            original_source_qualifications=dict(Counter(r['qualification'] for r in completed.values())),
            mutation_qualifications=dict(Counter(m['qualification'] for r in completed.values() for m in r['mutations'])),
            source_groups_with_pairs=sum(bool(v) for v in pair_rows.values()),plan_sha256=file_hash(REPORTS/'plan.json'),
            artifact_files_sha256={p.name:file_hash(p) for p in DATA.glob('*.jsonl')},
            execution_reports_sha256={p.name:file_hash(p) for p in REPORTS.glob('module-*.json')},
            source_files_sha256=files,human_preferences=False,training_admitted=False,model_updates=0,goal_achieved=False,
            scope='Supplemental complete original Python publisher-program execution feedback. Every source remains accounted for; qualification failures are not negative training labels. Original full-code/all30 replay requirements and genuine human feedback remain.')
        atomic_json(DATA/'manifest.json',final)
        atomic_json(REPORTS/'status.json',dict(status='all86 original Python modules accounted; candidate feedback complete',
            complete_passing_originals=len(positives),finite_assertion_pairs=final['finite_assertion_pairs'],
            training_admitted=False,model_updates=0,goal_achieved=False))
        print(json.dumps({k:final[k] for k in ['original_training_sources','complete_passing_originals','finite_assertion_pairs',
            'original_source_qualifications','mutation_qualifications']},indent=2))
    finally:suites.close()


if __name__=='__main__':main()
