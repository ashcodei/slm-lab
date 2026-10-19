"""Retain the complete v1 evidence and execute the one reviewed train helper case."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.execution_code_feedback_v1 import mutations,classify,sequence
from slm_lab.execution_java_training_suites_v2 import TrainingSuites
from slm_lab.exercise_java_runner import extract_files
from slm_lab.exercise_java_evaluation import question
from slm_lab.model import ModelConfig
from slm_lab.training import Encoder

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'data/execution-code-feedback-v1'
OLD_REPORTS=ROOT/'reports/execution-code-feedback-v1'
DATA=ROOT/'data/execution-code-feedback-v2'
REPORTS=ROOT/'reports/execution-code-feedback-v2'


def main():
    if DATA.exists() or REPORTS.exists():raise ValueError('Preserve prior v2 feedback evidence')
    if shutil.disk_usage(ROOT).free<26*1024**3+320*1024**2:raise ValueError('Fund all complete reports above26GiB')
    old=json.loads((OLD/'manifest.json').read_text());admission=json.loads((OLD/'admission.json').read_text())
    if not admission['passed'] or admission['candidate_manifest_sha256']!=file_hash(OLD/'manifest.json'):
        raise ValueError('Complete independently admitted parent feedback required')
    files={**admission['source_files_sha256'],str(OLD/'admission.json'):file_hash(OLD/'admission.json'),
        str(Path(__file__).resolve()):file_hash(__file__)}
    for name in ['src/slm_lab/execution_java_training_suites_v2.py','scripts/audit_execution_code_feedback_v2.py']:
        files[str(ROOT/name)]=file_hash(ROOT/name)
    guard=FileBindings(files)
    source=ROOT/'data/exercism-support-context-v3';rows=[r for r in map(json.loads,(source/'train.jsonl').open()) if r['language']=='Java']
    row=next(r for r in rows if r['id']=='exercism:java:tree-building')
    before=json.loads((OLD_REPORTS/'module-tree-building.json').read_text())
    if before['qualification']!='original_layout_or_environment_review':raise ValueError('Exact original helper qualification required')
    token_path=ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json';tok=Tokenizer.from_file(str(token_path))
    encoder=Encoder(tok,ModelConfig.load(ROOT/'configs/model.json'),source,448)
    suites=TrainingSuites(ROOT/'data/exercism-source-cache-v1/java')
    DATA.mkdir();REPORTS.mkdir()
    try:
        for path in OLD_REPORTS.glob('module-*.json'):
            if path.name!='module-tree-building.json':shutil.copy2(path,REPORTS/path.name)
        suite=suites.suite(row);original=suites.reference(row);prompt=encoder.prompt(question(row,suite))
        ids,labels=sequence(tok,prompt,row['response'])
        record=dict(id=row['id'],group_id=row['group_id'],split='train',language='Java',
            source_response_sha256=hashlib.sha256(row['response'].encode()).hexdigest(),mutations=[],
            prompt=prompt,original_response=row['response'],original_context_tokens=len(ids),original_target_tokens=sum(v!=-100 for v in labels),
            test_sha256=suite['test_sha256'],editor_sha256=suite['editor_sha256'],expected_solution_files=suite['expected_solution_files'],
            source_refinement='Original complete Record/TreeNode starters supplied unchanged; TreeNode differs from archived reference in access modifiers. Tests verify the actual provided original source.')
        reference=suites.run(original,suite);record.update(reference_execution=reference,qualification=classify(reference))
        atomic_json(REPORTS/'module-tree-building.json',record)
        if record['qualification']!='all_finite_published_cases_pass':raise ValueError('Reviewed original provided starters must actually pass all publisher cases')
        new_pairs=[]
        for number,mutant in enumerate(mutations(row['response'],row['id']),1):
            mids,mlabels=sequence(tok,prompt,mutant['response'])
            observed=suites.run(extract_files(mutant['response'],suite['expected_solution_files']),suite,reference['inventory'])
            label=classify(observed,reference['inventory'])
            entry=dict(number=number,**mutant,execution=observed,qualification=label,context_tokens=len(mids),target_tokens=sum(v!=-100 for v in mlabels))
            record['mutations'].append(entry)
            if label=='completed_published_assertion_counterexample':
                new_pairs.append(dict(id=row['id']+':mutation:'+str(number),source_id=row['id'],group_id=row['group_id'],split='train',language='Java',
                    prompt=prompt,chosen=row['response'],rejected=mutant['response'],chosen_sha256=record['source_response_sha256'],
                    rejected_sha256=mutant['response_sha256'],mutation=mutant['patch'],mutation_number=number,
                    chosen_context_tokens=len(ids),chosen_target_tokens=record['original_target_tokens'],rejected_context_tokens=len(mids),rejected_target_tokens=entry['target_tokens'],
                    original_test_sha256=suite['test_sha256'],original_editor_sha256=suite['editor_sha256'],registered_cases=reference['inventory']['cases'],
                    failed_cases=observed['inventory']['failed'],reference_report_sha256=reference['report_sha256'],mutant_report_sha256=observed['report_sha256'],
                    feedback_source='machine_execution_of_original_publisher_tests',human_preferences=False,
                    label_scope='Original source passes all complete publisher cases with original supplied helpers; the single-token mutant has a completed assertion failure. Finite machine execution only.'))
        atomic_json(REPORTS/'module-tree-building.json',record)
        grouped=defaultdict(list)
        for pair in map(json.loads,(OLD/'pairs.jsonl').open()):grouped[pair['source_id']].append(pair)
        if grouped[row['id']]:raise ValueError('Parent source was qualified and cannot already carry training pairs')
        grouped[row['id']]=new_pairs
        with (DATA/'pairs.jsonl').open('x') as out:
            for source_row in rows:
                for pair in grouped[source_row['id']]:out.write(json.dumps(pair,separators=(',',':'))+'\n')
        plan=json.loads((OLD_REPORTS/'plan.json').read_text());plan.update(created_at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files,parent_manifest_sha256=file_hash(OLD/'manifest.json'),parent_admission_sha256=file_hash(OLD/'admission.json'),
            source_revision=2,reviewed_helper_source=row['id'],parent_native_results_reused_only_when_byte_identical=True,
            only_tree_building_layout_changes=True,held_evaluation_runner_unchanged=True)
        atomic_json(REPORTS/'plan.json',plan)
        manifest=dict(old);counts=Counter(old['mutation_qualifications']);counts.update(m['qualification'] for m in record['mutations'])
        manifest.update(completed_at=datetime.now(timezone.utc).isoformat(),source_revision=2,
            original_module_qualifications={'all_finite_published_cases_pass':109},mutation_qualifications=dict(counts),
            source_modules_with_pairs=old['source_modules_with_pairs']+bool(new_pairs),pairs=old['pairs']+len(new_pairs),
            source_files_sha256=files,plan_sha256=file_hash(REPORTS/'plan.json'),
            artifact_files_sha256={'pairs.jsonl':file_hash(DATA/'pairs.jsonl')},
            execution_reports_sha256={p.name:file_hash(p) for p in REPORTS.glob('module-*.json')},
            parent_candidate_manifest_sha256=file_hash(OLD/'manifest.json'),parent_admission_sha256=file_hash(OLD/'admission.json'),
            training_admitted=False,model_updates=0,goal_achieved=False)
        guard.check(full=True);suites.guard.check(full=True)
        atomic_json(DATA/'manifest.json',manifest)
        atomic_json(REPORTS/'status.json',dict(status='all109original publisher references pass; independently audit combined feedback before training',
            original_modules=109,pairs=manifest['pairs'],training_admitted=False,model_updates=0,goal_achieved=False))
        print(json.dumps(dict(original_modules=109,newly_verified_tree_cases=len(reference['inventory']['cases']),
            new_tree_assertion_pairs=len(new_pairs),all_feedback_pairs=manifest['pairs']),indent=2))
    finally:suites.close()


if __name__=='__main__':main()
