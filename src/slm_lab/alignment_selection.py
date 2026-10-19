"""Matched local capability guards for feedback branches; not a success certificate."""
import math

from .code_tasks import TASKS
from .reward_gate import wilson

REPORT_FILES = {'code':'code-quality.json', 'code_instruction':'code-quality-instruction.json',
                'knowledge':'knowledge.json', 'math':'math.json', 'vision':'vision.json',
                'ocr':'ocr.json', 'objects':'objects.json', 'public_code':'public-code.json', 'cpp':'cpp-code.json'}
PROTOCOL_FIELDS = {
    'code':['suite_sha256','task_count','generation_budget','decoding','prompt_mode'],
    'code_instruction':['suite_sha256','task_count','generation_budget','decoding','prompt_mode'],
    'knowledge':['questions_sha256','protocol','generation_budget'],
    'math':['data_sha256','manifest_sha256','official_test_examples','evaluated_examples','full_official_test','generation_budget','protocol'],
    'vision':['data_sha256','image_size','visual_tokens','images'],
    'ocr':['data_sha256','image_size','visual_tokens','max_new_tokens','protocol'],
    'objects':['data_sha256','image_size','max_new_tokens','protocol'],
    'public_code':['data_sha256','official_sanitized_test_problems','supported_problems','excluded_problems','generation_budget','protocol'],
    'cpp':['task_count','generation_budget','prompt_mode'],
}
CASE_FIELDS = {
    'code':('results',['language','task','specification']),
    'code_instruction':('results',['language','task','specification']),
    'knowledge':('predictions',['id','domain','answer_index']),
    'math':('results',['id','expected_numerator','expected_denominator']),
    'vision':('samples',['id','domain','wrong_image_id','input_pixels_sha256','wrong_pixels_sha256']),
    'ocr':('results',['id','group_id','task','expected','input_pixels_sha256']),
    'objects':('results',['id','group_id','task','expected','wrong_image_id','input_pixels_sha256','wrong_pixels_sha256']),
    'public_code':('results',['id','task_id']),
    'cpp':('results',['task','specification','signature']),
}


def number(value, name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('Invalid/non-finite metric: '+name)
    return float(value)


def validate_collection(reports, families, require_provenance=True):
    digests = {reports[name]['checkpoint_sha256'] for name in families}
    if len(digests)!=1 or not next(iter(digests)):raise ValueError('Capability reports mix checkpoints')
    if require_provenance:
        first=reports[families[0]]['evaluation_provenance']
        if not first.get('files_sha256'):raise ValueError('Evaluation source provenance is missing')
        for name in families:
            if reports[name]['evaluation_provenance']!=first:raise ValueError('Capability reports mix implementations/runtimes')
            for key in ['inference_device_type','inference_precision','tokenizer_sha256']:
                if not reports[name].get(key):raise ValueError('Inference protocol metadata missing')
    return next(iter(digests))


def compare(baseline,candidate,preferences,families=None,require_provenance=True):
    families=list(REPORT_FILES) if families is None else list(families)
    if not families or len(set(families))!=len(families) or any(name not in REPORT_FILES for name in families):
        raise ValueError('Unknown or duplicated capability families')
    complete=set(families)==set(REPORT_FILES) and require_provenance
    before_sha=validate_collection(baseline,families,require_provenance)
    after_sha=validate_collection(candidate,families,require_provenance)
    checks={}

    def check(name,before,after,minimum=None,maximum=None):
        before,after=number(before,name),number(after,name);delta=after-before
        checks[name]={'baseline':before,'candidate':after,'delta':delta,
                      'passed':(minimum is None or delta>=minimum-1e-12) and (maximum is None or delta<=maximum+1e-12)}

    for name in families:
        old,new=baseline[name],candidate[name]
        fields=PROTOCOL_FIELDS[name]+(['evaluation_provenance','inference_device_type','inference_precision','tokenizer_sha256'] if require_provenance else [])
        for key in fields:
            if not require_provenance and key=='generation_budget' and name=='knowledge':continue
            if key not in old or key not in new or old[key]!=new[key]:raise ValueError('Incomparable '+name+' protocol: '+key)
        list_key,identity=CASE_FIELDS[name]
        if not require_provenance:identity=[key for key in identity if 'pixels_sha256' not in key]
        first=[tuple(row[key] for key in identity) for row in old[list_key]]
        second=[tuple(row[key] for key in identity) for row in new[list_key]]
        if not first or first!=second or len(set(first))!=len(first):raise ValueError('Incomparable or duplicate '+name+' cases')
        if complete:
            minimum={'code':24,'code_instruction':24,'knowledge':559,'math':1319,'vision':128,'ocr':170,'objects':164,'public_code':175,'cpp':12}[name]
            if len(first)<minimum:raise ValueError('Insufficient '+name+' assessment coverage')
            if name in ['code','code_instruction']:
                expected={(lang,task[0]) for lang in ['Python','JavaScript'] for task in TASKS}
                if {(row['language'],row['task']) for row in old[list_key]}!=expected:raise ValueError('Incomplete multilingual code task coverage')
            if name=='cpp' and {row['task'] for row in old[list_key]}!={task[0] for task in TASKS}:raise ValueError('Incomplete C++ task coverage')
            if name=='math' and (not old['full_official_test'] or len(first)!=old['official_test_examples']):raise ValueError('Math assessment must cover the full official test')
        if name in ['code','code_instruction','cpp']:
            for metric in ['syntax_pass_rate','functional_pass_at_1','efficiency_stress_pass_rate']:
                check(name+'_'+metric,old[metric],new[metric],minimum=0)
            losses=[]
            for a,b in zip(old['results'],new['results']):
                if a['candidate'].get('functional_passed',False) and not b['candidate'].get('functional_passed',False):
                    losses.append(a.get('language','C++')+':'+a['task'])
            checks[name+'_previously_correct_tasks']={'lost_tasks':losses,'passed':not losses}
            if name!='cpp':
                if set(old['by_language'])!=set(new['by_language']):raise ValueError('Code language coverage differs')
                for language in old['by_language']:
                    if old['by_language'][language]['tasks']!=new['by_language'][language]['tasks']:raise ValueError('Code language counts differ')
                    for metric in ['syntax_passed','functional_passed','efficiency_passed']:
                        check(name+'_'+language+'_'+metric,old['by_language'][language][metric],new['by_language'][language][metric],minimum=0)
            if name=='cpp':
                for key in ['version','compiler_sha256','archive_sha256']:
                    if old['toolchain'][key]!=new['toolchain'][key]:raise ValueError('Incomparable C++ compiler')
        elif name=='knowledge':
            if set(old['by_domain'])!=set(new['by_domain']):raise ValueError('Knowledge domain coverage differs')
            for metric in ['generated_accuracy','likelihood_accuracy']:
                check('knowledge_'+metric,old['overall'][metric],new['overall'][metric],minimum=-.02)
                for domain in old['by_domain']:
                    check('knowledge_'+domain+'_'+metric,old['by_domain'][domain][metric],new['by_domain'][domain][metric],minimum=-.025)
        elif name=='math':
            check('math_correctness',old['numeric_exact_accuracy'],new['numeric_exact_accuracy'],minimum=-.01)
            check('math_format',old['format_compliance'],new['format_compliance'],minimum=-.01)
        elif name=='public_code':
            for metric in ['syntax_rate_on_supported_subset','pass_at_1_on_supported_subset']:
                check('public_code_'+metric,old[metric],new[metric],minimum=0)
        elif name=='vision':
            check('caption_nll',old['proper_nll'],new['proper_nll'],maximum=.1)
            check('caption_specific_image_dependence',old['mismatch_nll_gain'],new['mismatch_nll_gain'],minimum=-.2*max(0,old['mismatch_nll_gain']))
            for metric in ['image_to_text_accuracy','text_to_image_accuracy']:
                check('caption_'+metric,old['caption_retrieval'][metric],new['caption_retrieval'][metric],minimum=0)
        else:
            if set(old['by_task'])!=set(new['by_task']):raise ValueError(name+' task coverage differs')
            accuracy='exact_match_accuracy' if name=='ocr' else 'correct_image_exact_accuracy'
            check(name+'_accuracy',old['overall'][accuracy],new['overall'][accuracy],minimum=0)
            for task in old['by_task']:
                a,b=old['by_task'][task],new['by_task'][task]
                if a['examples']!=b['examples']:raise ValueError('Visual task counts differ')
                check(name+'_'+task+'_accuracy',a[accuracy],b[accuracy],minimum=0)
                if name=='ocr':check('ocr_'+task+'_character_error',a['character_error_rate'],b['character_error_rate'],maximum=.05)
                else:
                    check('objects_'+task+'_nll',a['proper_nll'],b['proper_nll'],maximum=.1)
                    check('objects_'+task+'_specific_image_dependence',a['mismatch_nll_gain'],b['mismatch_nll_gain'],minimum=-.02)
    if preferences.get('task')!='policy_preference' or preferences['checkpoint_sha256']!=after_sha or preferences['reference_checkpoint_sha256']!=before_sha:
        raise ValueError('Preference evidence does not compare the candidate to this base')
    count,correct=preferences['examples'],preferences['correct']
    if type(count) is not int or type(correct) is not int:raise ValueError('Preference counts must be integers')
    bounds=wilson(correct,count)
    if len(preferences['sample_indices'])!=count or len(set(preferences['sample_indices']))!=count:raise ValueError('Duplicate/missing preference cases')
    if not preferences.get('data_sha256') or set(preferences.get('feedback_sources',[]))!={'human'}:raise ValueError('Human preference provenance missing')
    if require_provenance and preferences['evaluation_provenance']!=candidate[families[0]]['evaluation_provenance']:raise ValueError('Preference/capability implementation mismatch')
    if require_provenance:
        for key in ['tokenizer_sha256','inference_device_type','inference_precision']:
            if preferences.get(key)!=candidate[families[0]][key]:raise ValueError('Preference/capability inference protocol differs')
    checks['positive_preference_margin_lower_bound']={'lower_bound':bounds[0],'passed':bounds[0]>.5}
    margin=number(preferences['mean_reference_adjusted_margin'],'mean preference margin')
    checks['mean_preference_margin']={'mean':margin,'passed':margin>0}
    checks['preference_coverage']={'examples':count,'passed':count>=500 and preferences.get('truncated_sequences',0)==0}
    failures=[name for name,result in checks.items() if not result['passed']]
    return {'eligible_for_selection':complete and not failures,'evidence_complete':complete,
            'base_checkpoint_sha256':before_sha,'candidate_checkpoint_sha256':after_sha,
            'checks':checks,'failed_checks':failures,'preference_positive_margin_95ci':bounds,
            'scope':'Matched local tuning heuristics across declared capabilities, plus offline human-pair likelihood improvement; not fresh human judgement of generated responses.',
            'goal_achieved':False,'pending':['Fresh human own-policy comparisons','Broader languages/repository work and independent multimodal/reasoning tests','Matched high-end-model comparison']}
