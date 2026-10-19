"""Reconcile available native source reports without changing their protocol."""
from datetime import datetime,timezone
import json
import math
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/code-initialized-alignment-v1'
OUTPUT=ROOT/'reports/code-initialized-native-interim-v2.json'


def main():
    if OUTPUT.exists():raise ValueError('Preserve prior native report snapshots')
    hashes={};result={}
    def load(path):
        hashes[str(path.resolve())]=file_hash(path)
        return json.loads(path.read_text())
    def ratio(found,denominator,reported):
        if not math.isclose(found/denominator,reported,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('All original raw outcomes must reproduce the reported rate')
    for arm in ('initial','mixed_CE','mixed_reference_KL'):
        folder=REPORTS/arm;entry={};knowledge=load(folder/'knowledge.json');raw=knowledge['predictions'];n=len(raw)
        if n!=559 or n!=knowledge['overall']['questions']:raise ValueError('Complete original knowledge population required')
        generated=sum(r['generated_correct'] for r in raw);likelihood=sum(r['likelihood_correct'] for r in raw)
        if any((r['generated_index']==r['answer_index'])!=r['generated_correct'] or (r['likelihood_index']==r['answer_index'])!=r['likelihood_correct'] for r in raw):
            raise ValueError('Original answers and generated/likelihood outcomes differ')
        ratio(generated,n,knowledge['overall']['generated_accuracy']);ratio(likelihood,n,knowledge['overall']['likelihood_accuracy'])
        entry['knowledge']=dict(questions=n,generated_correct=generated,likelihood_correct=likelihood,reported_overall=knowledge['overall'])
        for name,required in [('code-completion',24),('code-instruction',24),('C++',12)]:
            code=load(folder/(name+'.json'));raw=code['results'];n=len(raw)
            if n!=required or n!=code['task_count']:raise ValueError('Every native coding task required')
            syntax=sum(r['candidate']['syntax_valid'] for r in raw);functional=sum(r['candidate']['functional_passed'] for r in raw)
            efficiency=sum(r['efficiency_stress_passed'] for r in raw)
            ratio(syntax,n,code['syntax_pass_rate']);ratio(functional,n,code['functional_pass_at_1']);ratio(efficiency,n,code['efficiency_stress_pass_rate'])
            entry[name]=dict(tasks=n,syntax_passes=syntax,functional_passes=functional,efficiency_passes=efficiency)
        ocr=load(folder/'ocr.json');raw=ocr['results'];n=len(raw);exact=sum(r['exact_match'] for r in raw)
        edits=sum(r['character_edits'] for r in raw);characters=sum(r['reference_characters'] for r in raw)
        ratio(exact,n,ocr['overall']['exact_match_accuracy']);ratio(edits,characters,ocr['overall']['character_error_rate'])
        if n!=170:raise ValueError('Every original OCR task required')
        entry['OCR']=dict(examples=n,exact_matches=exact,character_edits=edits,reference_characters=characters,character_error_rate=edits/characters)
        objects=load(folder/'objects.json');raw=objects['results'];n=len(raw)
        if n!=164:raise ValueError('Every proper/blank/wrong object control required')
        counts={k:sum(r[k] for r in raw) for k in ['proper_exact','blank_exact','wrong_exact']}
        for raw_key,metric in [('proper_exact','correct_image_exact_accuracy'),('blank_exact','blank_original_target_match'),('wrong_exact','wrong_image_original_target_match')]:
            ratio(counts[raw_key],n,objects['overall'][metric])
        entry['objects']=dict(examples=n,**counts)
        vision=load(folder/'vision.json');entry['caption_retrieval']=vision['caption_retrieval']
        public_files=[folder/f'public-{i}.json' for i in range(4)]
        if all(p.exists() for p in public_files):
            seen=set();syntax=functional=0
            for p in public_files:
                public=load(p);rows=public['results'];count=len(rows)
                if count!=public['supported_problems']:raise ValueError('Complete original public shard required')
                for row in rows:
                    if row['id'] in seen:raise ValueError('Public code shards overlap')
                    seen.add(row['id'])
                passed=sum(r['execution']['functional_passed'] for r in rows);valid=sum(r['execution']['syntax_valid'] for r in rows)
                ratio(passed,count,public['pass_at_1_on_supported_subset']);ratio(valid,count,public['syntax_rate_on_supported_subset'])
                syntax+=valid;functional+=passed
            if len(seen)!=175:raise ValueError('Every supported public program required')
            entry['public_code']=dict(supported_problems=len(seen),syntax_passes=syntax,functional_passes=functional,
                passing_program_execution_independently_replayed_by_this_reader=False)
        else:entry['public_code']=dict(complete=False)
        entry['native_math']=dict(complete=(folder/'math-complete.json').exists(),numerical_scores_reconciled_by_this_snapshot=False)
        result[arm]=entry
    hashes[str(Path(__file__).resolve())]=file_hash(__file__)
    for p,digest in hashes.items():
        if file_hash(p)!=digest:raise ValueError('Native snapshot source changed during readback')
    atomic_json(OUTPUT,dict(observed_at=datetime.now(timezone.utc).isoformat(),available_all_arm_native_results=result,
        original_raw_outcomes_and_report_arithmetic_reconciled=True,source_reports_sha256=hashes,
        neural_generation_independently_replayed=False,all_arm_public_and_math_comparison_finished=False,
        repeated_development_evaluations_not_certified_final_independence=True,official_scores_unchanged=True,
        no_candidate_promoted=True,model_updates=0,goal_achieved=False))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
