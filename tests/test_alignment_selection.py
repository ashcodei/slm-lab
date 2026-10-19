from copy import deepcopy
import pytest

from slm_lab.alignment_selection import compare
from slm_lab.code_tasks import TASKS


def evidence():
    provenance={'version':1,'files_sha256':{'model':'source'}}
    shared={'checkpoint_sha256':'base','evaluation_provenance':provenance,
            'inference_device_type':'cuda','inference_precision':'bf16','tokenizer_sha256':'tokenizer'}
    reports={}
    for name in ['code','code_instruction']:
        rows=[{'language':lang,'task':task[0],'specification':task[2],
               'candidate':{'syntax_valid':True,'functional_passed':False},'efficiency_stress_passed':False}
              for lang in ['Python','JavaScript'] for task in TASKS]
        reports[name]={**shared,'suite_sha256':'suite','task_count':24,'generation_budget':512,
                       'decoding':'greedy','prompt_mode':name,'results':rows,'syntax_pass_rate':1.,
                       'functional_pass_at_1':0.,'efficiency_stress_pass_rate':0.,
                       'by_language':{lang:{'tasks':12,'syntax_passed':12,'functional_passed':0,'efficiency_passed':0} for lang in ['Python','JavaScript']}}
    reports['cpp']={**shared,'task_count':12,'generation_budget':512,'prompt_mode':'instruction',
                    'syntax_pass_rate':1.,'functional_pass_at_1':0.,'efficiency_stress_pass_rate':0.,
                    'toolchain':{'version':'version','compiler_sha256':'compiler','archive_sha256':'archive'},
                    'results':[{'task':t[0],'specification':t[2],'signature':t[0],
                                'candidate':{'syntax_valid':True,'functional_passed':False}} for t in TASKS]}
    reports['math']={**shared,'data_sha256':'math','manifest_sha256':'manifest','official_test_examples':1319,
                     'evaluated_examples':1319,'full_official_test':True,'generation_budget':512,'protocol':'math',
                     'numeric_exact_accuracy':0.,'format_compliance':1.,
                     'results':[{'id':f'math-{i}','expected_numerator':1,'expected_denominator':1} for i in range(1319)]}
    reports['knowledge']={**shared,'questions_sha256':'knowledge','protocol':'knowledge','generation_budget':32,
                          'overall':{'generated_accuracy':.25,'likelihood_accuracy':.25},
                          'by_domain':{str(i):{'generated_accuracy':.25,'likelihood_accuracy':.25} for i in range(14)},
                          'predictions':[{'id':f'qa-{i}','domain':str(i%14),'answer_index':0} for i in range(559)]}
    reports['vision']={**shared,'data_sha256':'vision','image_size':448,'visual_tokens':196,'images':128,
                       'proper_nll':2.,'mismatch_nll_gain':.1,
                       'caption_retrieval':{'image_to_text_accuracy':.1,'text_to_image_accuracy':.1},
                       'samples':[{'id':f'photo-{i}','domain':'photos','wrong_image_id':f'photo-{(i+1)%128}',
                                   'input_pixels_sha256':f'pixels-{i}','wrong_pixels_sha256':f'pixels-{(i+1)%128}'} for i in range(128)]}
    ocr_rows=[];object_rows=[]
    for task,count in [('document_ocr',21),('document_region_ocr',21),('word_ocr',64),('region_ocr',64)]:
        ocr_rows.extend({'id':f'{task}-{i}','group_id':f'group-{task}-{i}','task':task,'expected':'text','input_pixels_sha256':f'pixels-{task}-{i}'} for i in range(count))
    reports['ocr']={**shared,'data_sha256':'ocr','image_size':448,'visual_tokens':196,'max_new_tokens':128,'protocol':'ocr',
                    'overall':{'exact_match_accuracy':0.},'results':ocr_rows,
                    'by_task':{task:{'examples':sum(r['task']==task for r in ocr_rows),'exact_match_accuracy':0.,'character_error_rate':1.} for task in {r['task'] for r in ocr_rows}}}
    for task,count in [('object_count',56),('object_region',60),('object_spatial',48)]:
        object_rows.extend({'id':f'{task}-{i}','group_id':f'group-{task}-{i}','task':task,'expected':'object',
                            'wrong_image_id':f'{task}-{(i+1)%count}','input_pixels_sha256':f'pixels-{task}-{i}',
                            'wrong_pixels_sha256':f'pixels-{task}-{(i+1)%count}'} for i in range(count))
    reports['objects']={**shared,'data_sha256':'objects','image_size':448,'max_new_tokens':32,'protocol':'objects',
                        'overall':{'correct_image_exact_accuracy':0.},'results':object_rows,
                        'by_task':{task:{'examples':sum(r['task']==task for r in object_rows),'correct_image_exact_accuracy':0.,
                                        'proper_nll':2.,'mismatch_nll_gain':.1} for task in {r['task'] for r in object_rows}}}
    reports['public_code']={**shared,'data_sha256':'public','official_sanitized_test_problems':257,'supported_problems':175,
                            'excluded_problems':82,'generation_budget':512,'protocol':'public',
                            'syntax_rate_on_supported_subset':1.,'pass_at_1_on_supported_subset':0.,
                            'results':[{'id':f'public-{i}','task_id':i} for i in range(175)]}
    candidate=deepcopy(reports)
    for report in candidate.values():report['checkpoint_sha256']='candidate'
    preferences={'task':'policy_preference','checkpoint_sha256':'candidate','reference_checkpoint_sha256':'base',
                 'examples':1000,'correct':600,'sample_indices':list(range(1000)),'data_sha256':'preferences',
                 'feedback_sources':['human'],'evaluation_provenance':provenance,'truncated_sequences':0,
                 'tokenizer_sha256':'tokenizer','inference_device_type':'cuda','inference_precision':'bf16',
                 'mean_reference_adjusted_margin':1.}
    return reports,candidate,preferences


def test_complete_matched_evidence_can_admit_a_branch_without_claiming_the_goal():
    result=compare(*evidence())
    assert result['eligible_for_selection'] and result['evidence_complete']
    assert result['goal_achieved'] is False


def test_average_code_gain_cannot_hide_losing_a_previously_correct_function():
    baseline,candidate,pref=evidence()
    baseline['code']['results'][0]['candidate']['functional_passed']=True
    baseline['code']['functional_pass_at_1']=1/24;baseline['code']['by_language']['Python']['functional_passed']=1
    for row in candidate['code']['results'][1:3]:row['candidate']['functional_passed']=True
    candidate['code']['functional_pass_at_1']=2/24;candidate['code']['by_language']['Python']['functional_passed']=2
    result=compare(baseline,candidate,pref)
    assert not result['eligible_for_selection']
    assert 'code_previously_correct_tasks' in result['failed_checks']


def test_knowledge_average_does_not_hide_one_domain_regression():
    baseline,candidate,pref=evidence();candidate['knowledge']['overall']['generated_accuracy']=.3
    candidate['knowledge']['by_domain']['0']['generated_accuracy']=.1
    assert 'knowledge_0_generated_accuracy' in compare(baseline,candidate,pref)['failed_checks']


def test_losing_one_visual_task_rejects_feedback_even_when_other_metrics_hold():
    baseline,candidate,pref=evidence();baseline['ocr']['by_task']['word_ocr']['exact_match_accuracy']=.1
    assert not compare(baseline,candidate,pref)['eligible_for_selection']


@pytest.mark.parametrize('change',['pixels','budget','checkpoint','provenance','duplicate','partial_math'])
def test_incomparable_or_incomplete_assessments_fail_closed(change):
    baseline,candidate,pref=evidence()
    if change=='pixels':candidate['objects']['results'][0]['input_pixels_sha256']='changed'
    elif change=='budget':candidate['math']['generation_budget']=1024
    elif change=='checkpoint':candidate['cpp']['checkpoint_sha256']='other'
    elif change=='provenance':candidate['vision']['evaluation_provenance']={'files_sha256':{'model':'other'}}
    elif change=='duplicate':
        for reports in [baseline,candidate]:reports['math']['results'][1]=deepcopy(reports['math']['results'][0])
    elif change=='partial_math':
        for reports in [baseline,candidate]:
            reports['math']['results']=reports['math']['results'][:64]
            reports['math'].update(evaluated_examples=64,full_official_test=False)
    with pytest.raises(ValueError):compare(baseline,candidate,pref)


def test_no_positive_preference_improvement_or_small_samples_cannot_promote():
    baseline,candidate,pref=evidence();pref['mean_reference_adjusted_margin']=0
    assert not compare(baseline,candidate,pref)['eligible_for_selection']
    pref['mean_reference_adjusted_margin']=1;pref.update(examples=100,correct=90,sample_indices=list(range(100)))
    assert not compare(baseline,candidate,pref)['eligible_for_selection']


def test_partial_legacy_diagnostic_cannot_select_a_policy():
    baseline,candidate,pref=evidence()
    result=compare(baseline,candidate,pref,families=['code','knowledge'],require_provenance=False)
    assert not result['evidence_complete'] and not result['eligible_for_selection']
