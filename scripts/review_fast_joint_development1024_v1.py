"""Reexecute all code outputs and read back matched raw development outcomes."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from fractions import Fraction
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'reports/fast-joint-development1024-readback-v1.json'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU score reader must hide all GPUs')
    if OUTPUT.exists():raise ValueError('Preserve earlier actual outcome readback')
    from slm_lab.code_quality import execute
    from slm_lab.code_tasks import TASKS,request
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.knowledge_evaluation import letter
    from slm_lab.math_corpus import final_answer
    from slm_lab.ocr_evaluation import transcription_scores
    from monitor_fast_joint_regression_v1 import scores,regression
    tasks={t[0]:t for t in TASKS};summaries={};bindings={};work=[];reports={}
    for arm in ['initial','step-1024']:
        folder=ROOT/'reports/fast-joint-v3'/arm;summary=scores(folder)
        if summary is None:raise ValueError('All matched available code/knowledge/math/OCR reports required')
        bindings.update(summary['reports_sha256']);summaries[arm]=summary
        code=json.loads((folder/'pulse-code-completion.json').read_text())
        for row in code['results']:work.append((arm,row))
        knowledge=json.loads((folder/'pulse-knowledge.json').read_text());items=knowledge['predictions']
        for row in items:
            predicted=letter(row['generated_response'],len(row['candidate_scores']))
            best=max(range(len(row['candidate_scores'])),key=lambda i:row['candidate_scores'][i])
            if (row['generated_index']!=predicted or row['generated_correct']!=(predicted==row['answer_index'])
                    or row['likelihood_index']!=best or row['likelihood_correct']!=(best==row['answer_index'])):
                raise ValueError('Every raw knowledge response/likelihood decision must reconstruct')
        maths=json.loads((folder/'pulse-math.json').read_text())
        for row in maths['results']:
            answer=final_answer(row['raw_response']);gold=Fraction(row['expected_numerator'],row['expected_denominator'])
            if row['correct']!=(answer==gold):raise ValueError('Every raw math response exact rational answer must reconstruct')
        if sum(r['correct'] for r in maths['results'])!=maths['correct']:raise ValueError('Math total differs')
        ocr=json.loads((folder/'pulse-ocr.json').read_text());characters=edits=0
        for row in ocr['results']:
            values=transcription_scores(row['expected'],row['generated'])
            if any(row[k]!=v for k,v in values.items()):raise ValueError('Every original OCR transcription score must reconstruct')
            characters+=values['reference_characters'];edits+=values['character_edits']
        if edits/characters!=ocr['overall']['character_error_rate']:raise ValueError('OCR error arithmetic differs')
        summaries[arm]['knowledge_likelihood_correct']=sum(r['likelihood_correct'] for r in items)
        summaries[arm]['knowledge_questions']=len(items);summaries[arm]['math_examples']=maths['evaluated_examples']
        summaries[arm]['OCR_examples']=len(ocr['results'])
        reports[arm]=dict(knowledge_ids=[r['id'] for r in items],math_ids=[r['id'] for r in maths['results']],
            OCR_ids=[r['id'] for r in ocr['results']])
    if reports['initial']!=reports['step-1024']:raise ValueError('Exact matched original evaluation items required')
    def replay(item):
        arm,row=item;actual=execute(request(tasks[row['task']],row['language'],row['code']))
        if any(actual.get(k,False)!=row['candidate'].get(k,False) for k in ['syntax_valid','functional_passed']):
            raise ValueError('Independent code reexecution changes actual syntax/functional verdict: '+arm+'/'+row['task'])
        return dict(arm=arm,language=row['language'],task=row['task'],actual_reexecution=actual)
    with ThreadPoolExecutor(max_workers=4) as pool:reexecutions=list(pool.map(replay,work))
    objects={}
    for arm in ['initial','step-1024']:
        path=ROOT/'reports/fast-joint-corrected-visual-CPU-v1'/arm/'objects.json'
        if path.exists():
            d=json.loads(path.read_text());rows=d['results'];objects[arm]=dict(examples=len(rows),
                proper=sum(r['proper_exact'] for r in rows),blank=sum(r['blank_exact'] for r in rows),wrong=sum(r['wrong_exact'] for r in rows),
                precision='cpu/fp32',separate_corrected_object_protocol=True)
            bindings[str(path)]=file_hash(path)
    atomic_json(OUTPUT,dict(observed_at=datetime.now(timezone.utc).isoformat(),matched_development_scores=summaries,
        all48code_generations_independently_reexecuted=True,actual_code_reexecutions=reexecutions,
        every_raw_knowledge_math_OCR_response_and_metric_reconstructed=True,corrected_CPU_object_scores=objects,
        first_checkpoint_large_broad_regression=regression(summaries['initial'],summaries['step-1024']),
        generation_budgets=dict(code=128,knowledge=32,math=256,OCR=128),
        separate_from_earlier512token_full_code_protocol=True,original_initial128token_functional_pass_already_present_before_learning=True,
        not_independent_neural_generation_replay_or_full_benchmarks=True,source_files_sha256=bindings,
        reader_sha256=file_hash(__file__),candidate_promoted=False,goal_achieved=False))
    print(json.dumps(dict(scores={a:{k:v for k,v in s.items() if k not in ('protocol','reports_sha256')} for a,s in summaries.items()},corrected_CPU_objects=objects)),flush=True)


if __name__=='__main__':main()
