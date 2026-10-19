"""Recount retained native outputs; require matching inputs and protocols."""
from collections import defaultdict
import math

from .knowledge_evaluation import letter
from .ocr_evaluation import normalized,transcription_scores

NAMES=('code-completion.json','code-instruction.json','public-Python.json','C++.json',
    'knowledge.json','ocr.json','objects.json','vision.json')


def near(actual,expected):
    if not math.isfinite(actual) or not math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-9):
        raise ValueError('Reported native aggregate differs from actual retained outcomes')


def summarize(report,name,knowledge_rows=None):
    if name not in NAMES:raise ValueError('Known original native assessment required')
    if name in NAMES[:4]:
        rows=report['results'];n=len(rows);public=name=='public-Python.json'
        results=[r['execution'] if public else r['candidate'] for r in rows]
        for result in results:
            if result.get('functional_passed') and (not result.get('syntax_valid')
                    or result.get('total_cases',0)<1 or result.get('passed_cases')!=result['total_cases']):
                raise ValueError('No functional success without every actual finite native case')
        syntax=sum(bool(x.get('syntax_valid')) for x in results)
        functional=sum(bool(x.get('functional_passed')) for x in results)
        near(report['syntax_rate_on_supported_subset'] if public else report['syntax_pass_rate'],syntax/n)
        near(report['pass_at_1_on_supported_subset'] if public else report['functional_pass_at_1'],functional/n)
        outcome=dict(tasks=n,syntax=syntax,functional=functional)
        if not public:
            efficiency=sum(bool(r['efficiency_stress_passed']) for r in rows)
            if any(r['efficiency_stress_passed'] and not x.get('functional_passed') for r,x in zip(rows,results)):
                raise ValueError('No empirical efficiency pass without actual functional success')
            near(report['efficiency_stress_pass_rate'],efficiency/n);outcome['empirical_efficiency']=efficiency
        if name.startswith('code-'):
            outcome['by_language']={language:dict(tasks=sum(r['language']==language for r in rows),
                syntax=sum(r['language']==language and bool(r['candidate'].get('syntax_valid')) for r in rows),
                functional=sum(r['language']==language and bool(r['candidate'].get('functional_passed')) for r in rows))
                for language in sorted({r['language'] for r in rows})}
        return outcome
    if name=='knowledge.json':
        rows=report['predictions'];groups=defaultdict(list)
        if report['skipped']:raise ValueError('Matched complete knowledge assessment cannot skip original questions')
        if knowledge_rows is None or len(rows)!=len(knowledge_rows):raise ValueError('Every actual original domain question required')
        for row,original in zip(rows,knowledge_rows):
            scores=row['candidate_scores'];gold=int(original['answer_index'])
            if (row['id']!=original['id'] or row['domain']!=original['domain'] or row['answer_index']!=gold
                    or len(scores)!=len(original['options']) or not all(math.isfinite(x) for x in scores)):
                raise ValueError('Exact original question/answer/options and finite scores required')
            generated=letter(row['generated_response'],len(scores));choice=max(range(len(scores)),key=lambda i:scores[i])
            if (generated!=row['generated_index'] or choice!=row['likelihood_index']
                    or row['generated_correct']!=(generated==gold) or row['likelihood_correct']!=(choice==gold)):
                raise ValueError('Native answer text and likelihood scores must reproduce every domain outcome')
            groups[row['domain']].append(row)
        def summary(items):
            return dict(questions=len(items),generated=sum(r['generated_correct'] for r in items),
                likelihood=sum(r['likelihood_correct'] for r in items),
                format_compliant=sum(r['generated_index'] is not None for r in items))
        result=summary(rows)
        near(report['overall']['generated_accuracy'],result['generated']/len(rows))
        near(report['overall']['likelihood_accuracy'],result['likelihood']/len(rows))
        result['by_domain']={domain:summary(items) for domain,items in sorted(groups.items())}
        return result
    if name=='ocr.json':
        rows=report['results'];targets=sum(r['target_tokens'] for r in rows)
        for row in rows:
            expected=transcription_scores(row['expected'],row['generated'])
            if any(row[k]!=v for k,v in expected.items()):raise ValueError('Every actual OCR transcription score must reconstruct')
        exact=sum(r['exact_match'] for r in rows)
        cer=sum(r['character_edits'] for r in rows)/sum(r['reference_characters'] for r in rows)
        wer=sum(r['word_edits'] for r in rows)/sum(r['reference_words'] for r in rows)
        gain=sum((r['blank_nll']-r['proper_nll'])*r['target_tokens'] for r in rows)/targets
        for key,value in [('exact_match_accuracy',exact/len(rows)),('character_error_rate',cer),('word_error_rate',wer),('blank_image_nll_gain',gain)]:
            near(report['overall'][key],value)
        return dict(examples=len(rows),exact=exact,character_error_rate=cer,word_error_rate=wer,blank_image_nll_gain=gain)
    if name=='objects.json':
        rows=report['results'];result=dict(examples=len(rows));targets=sum(r['target_tokens'] for r in rows)
        for arm in ['proper','blank','wrong']:
            for row in rows:
                if row[arm+'_exact']!=(normalized(row[arm+'_generated']).casefold()==normalized(row['expected']).casefold()):
                    raise ValueError('Every proper/blank/wrong image answer must reconstruct')
            result[arm+'_original_target_matches']=sum(r[arm+'_exact'] for r in rows)
        for arm,key in [('proper','correct_image_exact_accuracy'),('blank','blank_original_target_match'),('wrong','wrong_image_original_target_match')]:
            near(report['overall'][key],result[arm+'_original_target_matches']/len(rows))
        result['proper_nll']=sum(r['proper_nll']*r['target_tokens'] for r in rows)/targets
        result['wrong_image_nll_gain']=sum((r['wrong_nll']-r['proper_nll'])*r['target_tokens'] for r in rows)/targets
        near(report['overall']['proper_nll'],result['proper_nll'])
        near(report['overall']['mismatch_nll_gain'],result['wrong_image_nll_gain'])
        return result
    rows=report['samples'];n=len(rows);targets=sum(r['target_tokens'] for r in rows)
    if report['images']!=n or report['target_tokens']!=targets:raise ValueError('Every actual caption source/target count required')
    proper=sum(r['proper_nll']*r['target_tokens'] for r in rows)/targets
    blank=sum((r['blank_nll']-r['proper_nll'])*r['target_tokens'] for r in rows)/targets
    wrong=sum((r['mismatched_nll']-r['proper_nll'])*r['target_tokens'] for r in rows)/targets
    for key,value in [('proper_nll',proper),('blank_nll_gain',blank),('mismatch_nll_gain',wrong)]:near(report[key],value)
    retrieval=report['caption_retrieval'];hits={}
    for direction in ['image_to_text','text_to_image']:
        value=retrieval[direction+'_accuracy']*n
        near(value,round(value));hits[direction+'_reported_hits']=round(value)
    return dict(images=n,proper_nll=proper,blank_image_nll_gain=blank,wrong_image_nll_gain=wrong,**hits,
        retrieval_from_native_report_similarity_matrix_not_retained=True)


def matched(before,after,name):
    fields=['tokenizer_sha256','evaluation_provenance','inference_device_type','inference_precision']
    fields+=['generation_budget','prompt_mode','suite_sha256','data_sha256','questions_sha256','contamination_audit_sha256',
        'max_new_tokens','image_size','visual_tokens','protocol']
    for key in fields:
        if before.get(key)!=after.get(key):raise ValueError('Native assessment protocol/input mismatch: '+key)
    field='predictions' if name=='knowledge.json' else 'samples' if name=='vision.json' else 'results'
    keys=(['id','domain','answer_index'] if name=='knowledge.json' else
        ['id','group_id','task','expected','input_pixels_sha256','wrong_pixels_sha256','wrong_image_id','target_tokens']
        if name in ['ocr.json','objects.json','vision.json'] else ['id','task_id','task','language','specification','signature'])
    a=[{k:r[k] for k in keys if k in r} for r in before[field]]
    b=[{k:r[k] for k in keys if k in r} for r in after[field]]
    if a!=b:raise ValueError('Every original source row/target/image intervention must match before and after')
    if len(a)!=len({tuple(sorted(x.items())) for x in a}):raise ValueError('Duplicate native source identities cannot count twice')
    return True
