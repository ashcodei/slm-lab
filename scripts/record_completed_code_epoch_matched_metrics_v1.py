"""Reconstruct the completed7558-epoch local metrics without claiming gains."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/code-first-continuation-v3'
OUT=ROOT/'reports/completed-code-epoch-matched-metrics-v1'


def read(p):return json.loads(Path(p).read_text())


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh completed actual metric-source reconstruction required')
    trained=read(SOURCE/'training-complete.json');done=read(SOURCE/'complete.json')
    if trained['actual_updates']!=7558 or not done['original_local_code_and_broad_assessments_complete']:
        raise ValueError('The actual complete source epoch and all native comparisons must finish')
    replay_path=ROOT/'reports/code-first-complete-native-program-reexecution-CPU-v1/complete.json';replay=read(replay_path)
    if (not replay['complete'] or not replay['all96_original_decoded_program_bytes_reconstructed_and_fresh_native_verdicts_match']
            or not replay['all48_original_Python_programs_additionally_run_in_ordinary_stdlib_protocol']):
        raise ValueError('Every actual native decoded program and ordinary Python case execution must already be complete')
    inventory=[SOURCE/'plan.json',SOURCE/'complete.json',SOURCE/'training-complete.json',replay_path,Path(__file__)];reports={}
    for name,identity in [('knowledge','questions_sha256'),('math','data_sha256'),('ocr','data_sha256'),('objects','data_sha256'),('vision','data_sha256')]:
        initial=read(SOURCE/'initial'/(name+'.json'));final=read(SOURCE/'trained'/(name+'.json'))
        common=[identity,'inference_device_type','inference_precision','tokenizer_sha256']
        if name in {'knowledge','math'}:common.append('generation_budget')
        if name in {'ocr','objects'}:common.extend(['image_size','max_new_tokens'])
        if name=='vision':common.extend(['image_size','images','target_tokens'])
        if any(initial[k]!=final[k] for k in common):raise ValueError('Original input population/budget/runtime must match for '+name)
        inventory.extend([SOURCE/arm/(name+'.json') for arm in ['initial','trained']]);reports[name]=[initial,final]
    metrics={}
    def counts(name,field,key):
        values=[]
        for report in reports[name]:
            rows=report[field];values.append(dict(correct=sum(bool(r[key]) for r in rows),total=len(rows)))
        return dict(initial=values[0],trained=values[1])
    metrics['code_functional']={arm:dict(correct=replay['actual_matched_endpoint_totals'][arm]['functional'],total=48) for arm in ['initial','trained']}
    metrics['code_syntax']={arm:dict(correct=replay['actual_matched_endpoint_totals'][arm]['syntax'],total=48) for arm in ['initial','trained']}
    metrics['ordinary_Python_functional']={arm:dict(correct=replay['actual_matched_endpoint_totals'][arm]['ordinary_Python_functional'],total=24) for arm in ['initial','trained']}
    metrics['knowledge_generated']=counts('knowledge','predictions','generated_correct')
    metrics['knowledge_likelihood_choice']=counts('knowledge','predictions','likelihood_correct')
    metrics['math_numeric_exact']=counts('math','results','correct')
    metrics['OCR_native_exact']=counts('ocr','results','exact_match')
    metrics['object_native_exact']=counts('objects','results','proper_exact')
    for arm,index in [('initial',0),('trained',1)]:
        k=reports['knowledge'][index]
        if metrics['knowledge_generated'][arm]['correct']/56!=k['overall']['generated_accuracy'] or len(k['predictions'])!=56:
            raise ValueError('All original knowledge predictions and actual count must reconstruct')
        math=reports['math'][index]
        if metrics['math_numeric_exact'][arm]['correct']!=math['correct'] or len(math['results'])!=16:raise ValueError('Every original math outcome required')
        ocr=reports['ocr'][index]
        if metrics['OCR_native_exact'][arm]['correct']/20!=ocr['overall']['exact_match_accuracy'] or len(ocr['results'])!=20:raise ValueError('Every original OCR outcome required')
        obj=reports['objects'][index]
        if metrics['object_native_exact'][arm]['correct']/15!=obj['overall']['correct_image_exact_accuracy'] or len(obj['results'])!=15:raise ValueError('Every original object outcome required')
    for field in ['image_to_text_accuracy','text_to_image_accuracy']:
        values={}
        for arm,index in [('initial',0),('trained',1)]:
            report=reports['vision'][index];score=report['caption_retrieval'][field]*report['images']
            if report['images']!=32 or not score.is_integer():raise ValueError('All original32caption candidates and integer retrieval outcomes required')
            values[arm]=dict(correct=int(score),total=32)
        metrics['photo_'+field]=values
    loss0=read(SOURCE/'held-packed-initial.json');loss1=read(SOURCE/'held-packed-7558.json')
    for key in ['selected_original_positions','complete_causal_targets','original_manifest_sha256']:
        if loss0[key]!=loss1[key]:raise ValueError('Exact matched original held loss windows and targets required')
    loss=dict(initial_NLL=loss0['token_mean_NLL'],trained_NLL=loss1['token_mean_NLL'],
        relative_reduction=1-loss1['token_mean_NLL']/loss0['token_mean_NLL'],windows=128,original_targets=524160,
        producer_neural_observations_no_fresh_loss_recomputation=True,not_generated_correctness=True)
    inventory.extend([SOURCE/'held-packed-initial.json',SOURCE/'held-packed-7558.json']);OUT.mkdir()
    atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,actual_matched_metrics=metrics,same_protocol_held_loss=loss,
        source_files_sha256={str(p):file_hash(p) for p in inventory},local_development_samples_not_full_official_QA_AI2D_MBPP_benchmarks=True,
        only_code_programs_freshly_reexecuted_other_neural_observations_not_newly_regenerated=True,
        original_generated_token_actions_not_retained_by_GPU_producer=True,meaningful_all_round_generated_capability_gain_not_established=True,
        model_updates=0,candidate_promoted=False,goal_achieved=False))
    lines=['| Local development check | Initial | After 7,558 updates |','|---|---:|---:|']
    labels=dict(code_functional='Correct code',code_syntax='Syntactically valid code',ordinary_Python_functional='Correct code in ordinary Python',
        knowledge_generated='Knowledge: generated answer',knowledge_likelihood_choice='Knowledge: choice likelihood',math_numeric_exact='Math: exact numeric answer',
        OCR_native_exact='OCR: exact native text',object_native_exact='Object questions: exact native answer',
        photo_image_to_text_accuracy='Photo to caption retrieval',photo_text_to_image_accuracy='Caption to photo retrieval')
    for name,values in metrics.items():
        before,after=values['initial'],values['trained'];lines.append(f'| {labels[name]} | {before["correct"]}/{before["total"]} | {after["correct"]}/{after["total"]} |')
    (OUT/'metrics.md').write_text('# Completed code epoch: matched local metrics\n\n'+'\n'.join(lines)+'\n\n'
        f'Held packed-code NLL: {loss["initial_NLL"]:.6f} → {loss["trained_NLL"]:.6f} ({100*loss["relative_reduction"]:.2f}% lower), on the same 128 windows and 524,160 targets. '
        'These are producer loss observations. Lower token loss has not established generated coding or broad capability gains.\n\n'
        'All 96 decoded code programs were freshly rerun on the unchanged native cases, and all 48 Python programs were also rerun in the ordinary stdlib protocol. '
        'Other scores are reconstructed from the completed producer observations, without new neural generation. These small local samples are separate from the full QA, AI2D and MBPP assessments still queued. '
        'The original GPU producer did not retain generated token-action/EOS journals, so those are not certified here.\n')
    print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_matched_metrics=metrics,same_protocol_held_loss=loss)),flush=True)


if __name__=='__main__':main()
