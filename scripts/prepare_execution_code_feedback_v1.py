"""Execute all109 original Java training modules and four source mutations each."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.execution_code_feedback_v1 import mutations,classify,sequence
from slm_lab.exercise_java_runner import PublishedSuites,extract_files
from slm_lab.exercise_java_evaluation import question
from slm_lab.training import Encoder
from slm_lab.model import ModelConfig

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/execution-code-feedback-v1'
REPORTS=ROOT/'reports/execution-code-feedback-v1'
SOURCE=ROOT/'data/exercism-support-context-v3'
TOKEN=ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json'
MAX_REPORT=256*1024**2
MAX_DATA=64*1024**2


def main():
    if DATA.exists() or REPORTS.exists():raise ValueError('Preserve prior execution feedback; no implicit restart')
    if shutil.disk_usage(ROOT).free<26*1024**3+MAX_REPORT+MAX_DATA:
        raise ValueError('Fund all complete execution reports above26GiB reserve')
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    for name in ['train.jsonl','valid.jsonl']:
        if file_hash(SOURCE/name)!=manifest['files_sha256'][name]:raise ValueError('Original sealed publisher source differs')
    train=list(map(json.loads,(SOURCE/'train.jsonl').open()));valid=list(map(json.loads,(SOURCE/'valid.jsonl').open()))
    if ({r['group_id'] for r in train}&{r['group_id'] for r in valid}):raise ValueError('Original train/validation groups must remain disjoint')
    rows=[r for r in train if r['language']=='Java']
    if len(rows)!=109 or len({r['id'] for r in rows})!=109 or any(r['split']!='train' or r['license']!='MIT' for r in rows):
        raise ValueError('All109 original licensed sealed Java training modules required')
    suites=PublishedSuites(ROOT/'data/exercism-source-cache-v1/java')
    spec_cache=ROOT/'data/exercism-source-cache-v1/problem-specifications'
    spec_source=json.loads((spec_cache/'source.json').read_text())
    if file_hash(spec_cache/'source.zip')!=spec_source['archive_sha256']:
        raise ValueError('Original problem-specifications archive differs')
    spec_archive=zipfile.ZipFile(spec_cache/'source.zip')
    spec_prefix=spec_archive.namelist()[0].split('/')[0]+'/'
    paths=[Path(__file__),SOURCE/'manifest.json',SOURCE/'train.jsonl',SOURCE/'valid.jsonl',TOKEN,
        ROOT/'configs/model.json',ROOT/'src/slm_lab/execution_code_feedback_v1.py',
        ROOT/'src/slm_lab/exercise_java_evaluation.py',ROOT/'src/slm_lab/training.py',ROOT/'src/slm_lab/model.py',
        ROOT/'reports/execution-code-feedback-v1-focused-tests.log',
        spec_cache/'source.json',spec_cache/'source.zip']
    files={**{str(p):digest for p,digest in suites.guard.files.items()},**{str(p):file_hash(p) for p in paths}}
    guard=FileBindings(files)
    tokenizer=Tokenizer.from_file(str(TOKEN));encoder=Encoder(tokenizer,ModelConfig.load(ROOT/'configs/model.json'),SOURCE,448)
    if manifest['tokenizer_sha256']!=file_hash(TOKEN):raise ValueError('Original own BPE required')
    # Verify every declared original byte before treating layout/runtime errors
    # as qualifications. A source-integrity mismatch must abort the whole job.
    for row in rows:
        for item in row['source_files']+row['published_test_artifacts']:
            repository=item.get('repository','exercism/java')
            if repository=='exercism/java':raw=suites.read(item['path'])
            elif repository=='exercism/problem-specifications':
                if (Path(item['path']).is_absolute() or '..' in Path(item['path']).parts
                        or '/blob/'+spec_source['revision']+'/' not in item['url']):
                    raise ValueError('Exact original canonical-data repository/revision/path required')
                raw=spec_archive.read(spec_prefix+item['path'])
            else:raise ValueError('Unknown original source repository requires explicit resolution')
            if hashlib.sha256(raw).hexdigest()!=item['sha256']:
                raise ValueError('Original publisher source/test byte mismatch')
    spec_archive.close()
    DATA.mkdir();REPORTS.mkdir()
    atomic_json(REPORTS/'plan.json',dict(format='own-public-Java-execution-feedback-plan-v1',
        created_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        source_population=[dict(id=r['id'],group_id=r['group_id'],language=r['language'],split=r['split']) for r in rows],
        complete_original_training_modules=109,mutations_per_fitting_passing_source=4,maximum_CPU_workers=4,
        seed=930031,maximum_report_bytes=MAX_REPORT,maximum_data_bytes=MAX_DATA,minimum_remaining_GiB=26,
        only_original_solution_operator_or_literal_changed=True,comments_strings_tests_support_unchanged=True,
        complete_original_question_target_and_mutated_response_EOS_preserved=True,
        compile_runtime_inventory_failures_not_preference_labels=True,
        labels='Finite complete publisher assertion failures versus original passing source only. Machine feedback, not human preferences or whole-program correctness.',
        original_positive_populations_unchanged=True,training_admitted=False,model_updates=0,goal_achieved=False))
    def guard_all(full=False):
        guard.check(full=full)
        if shutil.disk_usage(ROOT).free<26*1024**3:raise ValueError('Original persistent reserve must remain funded')
        if sum(p.stat().st_size for p in REPORTS.rglob('*') if p.is_file())>MAX_REPORT:
            raise ValueError('Execution report budget exceeded')
    def execute(row):
        guard_all()
        record=dict(id=row['id'],group_id=row['group_id'],split='train',language='Java',
            source_response_sha256=hashlib.sha256(row['response'].encode()).hexdigest(),mutations=[])
        try:
            suite=suites.suite(row);original=suites.reference(row)
        except (ValueError,KeyError,StopIteration) as error:
            guard_all()
            return dict(**record,qualification='original_layout_or_environment_review',detail=str(error)),[]
        prompt=encoder.prompt(question(row,suite))
        try:ids,labels=sequence(tokenizer,prompt,row['response'])
        except ValueError:
            return dict(**record,qualification='whole_original_extended_context_exceeds4096',prompt=prompt),[]
        record.update(prompt=prompt,original_response=row['response'],original_context_tokens=len(ids),
            original_target_tokens=sum(y!=-100 for y in labels),test_sha256=suite['test_sha256'],editor_sha256=suite['editor_sha256'],
            expected_solution_files=suite['expected_solution_files'])
        measured=suites.run(original,suite);qualification=classify(measured)
        record.update(reference_execution=measured,qualification=qualification)
        if qualification!='all_finite_published_cases_pass':return record,[]
        pairs=[]
        for number,mutant in enumerate(mutations(row['response'],row['id']),1):
            entry=dict(number=number,**mutant)
            try:mid,mlabels=sequence(tokenizer,prompt,mutant['response'])
            except ValueError:
                record['mutations'].append(dict(**entry,qualification='complete_mutated_response_exceeds4096'));continue
            observed=suites.run(extract_files(mutant['response'],suite['expected_solution_files']),suite,measured['inventory'])
            label=classify(observed,measured['inventory'])
            entry.update(execution=observed,qualification=label,context_tokens=len(mid),target_tokens=sum(y!=-100 for y in mlabels))
            record['mutations'].append(entry)
            if label=='completed_published_assertion_counterexample':
                pairs.append(dict(id=row['id']+':mutation:'+str(number),source_id=row['id'],group_id=row['group_id'],
                    split='train',language='Java',prompt=prompt,chosen=row['response'],rejected=mutant['response'],
                    chosen_sha256=record['source_response_sha256'],rejected_sha256=mutant['response_sha256'],
                    mutation=mutant['patch'],mutation_number=number,
                    chosen_context_tokens=len(ids),chosen_target_tokens=record['original_target_tokens'],
                    rejected_context_tokens=len(mid),rejected_target_tokens=entry['target_tokens'],
                    original_test_sha256=suite['test_sha256'],original_editor_sha256=suite['editor_sha256'],
                    registered_cases=measured['inventory']['cases'],failed_cases=observed['inventory']['failed'],
                    reference_report_sha256=measured['report_sha256'],mutant_report_sha256=observed['report_sha256'],
                    feedback_source='machine_execution_of_original_publisher_tests',human_preferences=False,
                    label_scope='Original source passes every complete publisher case; one-token mutant completes the identical full case inventory with a retained assertion failure. Finite tests only.'))
        return record,pairs
    completed={};all_pairs={}
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(execute,row):row for row in rows}
            for future in as_completed(futures):
                row=futures[future];record,pairs=future.result();completed[row['id']]=record;all_pairs[row['id']]=pairs
                atomic_json(REPORTS/('module-'+row['exercise']+'.json'),record)
                atomic_json(REPORTS/'status.json',dict(status='executing original training publisher suites and bounded source mutations',
                    updated_at=datetime.now(timezone.utc).isoformat(),completed_modules=len(completed),expected_modules=109,
                    finite_assertion_pairs=sum(map(len,all_pairs.values())),model_updates=0,training_admitted=False,goal_achieved=False))
                print('original_training_modules',len(completed),'of109;verified_assertion_pairs',sum(map(len,all_pairs.values())),flush=True)
        guard_all(True)
        if set(completed)!={r['id'] for r in rows}:raise ValueError('Every original training module must remain accounted for')
        with (DATA/'pairs.jsonl').open('x') as out:
            for row in rows:
                for pair in all_pairs[row['id']]:out.write(json.dumps(pair,separators=(',',':'))+'\n')
        if (DATA/'pairs.jsonl').stat().st_size>MAX_DATA:raise ValueError('Full pair metadata budget exceeded')
        counts=Counter(r['qualification'] for r in completed.values())
        mutation_counts=Counter(m['qualification'] for r in completed.values() for m in r['mutations'])
        final=dict(format='own-published-Java-execution-feedback-candidate-v1',completed_at=datetime.now(timezone.utc).isoformat(),
            original_complete_modules=109,original_module_qualifications=dict(counts),mutation_qualifications=dict(mutation_counts),
            source_modules_with_pairs=sum(bool(all_pairs[r['id']]) for r in rows),pairs=sum(map(len,all_pairs.values())),
            tokenizer_sha256=file_hash(TOKEN),original_source_manifest_sha256=file_hash(SOURCE/'manifest.json'),
            plan_sha256=file_hash(REPORTS/'plan.json'),artifact_files_sha256={'pairs.jsonl':file_hash(DATA/'pairs.jsonl')},
            execution_reports_sha256={p.name:file_hash(p) for p in REPORTS.glob('module-*.json')},
            source_files_sha256=files,training_admitted=False,human_preference_labels=False,model_updates=0,goal_achieved=False,
            ordinary_full_code_and_original_source_populations_unchanged=True,
            scope='Additional candidate finite-execution feedback on all original Java training modules. Other languages and existing complete code/replay sources remain required; this does not replace them or admit a new final production source.')
        atomic_json(DATA/'manifest.json',final)
        atomic_json(REPORTS/'status.json',dict(status='all109 original Java training modules accounted; candidate finite-execution feedback prepared',
            completed_modules=109,pairs=final['pairs'],training_admitted=False,model_updates=0,goal_achieved=False))
        print(json.dumps({k:final[k] for k in ['original_complete_modules','original_module_qualifications','mutation_qualifications','source_modules_with_pairs','pairs']},indent=2))
    finally:
        suites.close()


if __name__=='__main__':main()
