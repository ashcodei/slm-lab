"""Public implementation contrasts qualified by fixed numerical specifications.

The chosen program contains an unchanged original public alternative and, when
needed, a mechanical positional forwarding alias. Rejected code is the whole
original public implementation that failed a fresh numerical case. Neither is
a sampled model rollout or human ranking. Finite qualification is explicit.
"""
import hashlib

from .function_semantic_stress_v1 import DONORS,cases,digest,matches,public_alternative

FORMAT='public-original-implementation-semantic-source-contrast-v1'


def make(row,donor,negative,tokenizer):
    selected=cases(row);name=row['source_candidate']['function']
    if (name not in DONORS or negative.get('program_sha256')!=hashlib.sha256(row['response'].encode()).hexdigest()
            or negative.get('all_cases_observed') is not True or negative.get('cases')!=len(selected)
            or len(negative.get('raw_results',[]))!=len(selected)):
        raise ValueError('Every actual complete original negative case observation must be present')
    checks=[matches(case['expected'],raw) for case,raw in zip(selected,negative['raw_results'])]
    if negative.get('case_passes')!=checks or negative.get('all_cases_passed')!=all(checks) or all(checks):
        raise ValueError('Only an actual original mathematical failure may enter a repair contrast')
    position=next(i for i,v in enumerate(checks) if not v)
    case=selected[position];raw=negative['raw_results'][position]
    if case['arguments_not_in_original_public_examples'] is not True:raise ValueError('The failure must occur on fresh arguments')
    program=public_alternative(row,donor)
    prompt=('Repair this original public Python implementation according to its documentation. '
        'Return the complete corrected implementation and any needed helpers.\n\n'+row['prompt']+
        '\n\nOriginal implementation with a verified semantic failure:\n```python\n'+row['response']+
        '\n```\n\nAdditional numerical counterexample:\n'+name+'(*'+repr(case['args'])+')\nExpected: '+repr(case['expected'])+
        '\nObserved stdout: '+repr(raw['stdout'])+'\nObserved exception: '+repr(raw['exception_only'])+'\n')
    prefix=[1]+tokenizer.encode('User: '+prompt+'\nAssistant:',add_special_tokens=False).ids
    chosen=tokenizer.encode(program,add_special_tokens=False).ids+[2]
    rejected=tokenizer.encode(row['response'],add_special_tokens=False).ids+[2]
    if (max(len(prefix)+len(chosen),len(prefix)+len(rejected))>4096
            or tokenizer.decode(prefix[1:],skip_special_tokens=False)!='User: '+prompt+'\nAssistant:'
            or tokenizer.decode(chosen[:-1],skip_special_tokens=False)!=program
            or tokenizer.decode(rejected[:-1],skip_special_tokens=False)!=row['response']
            or any(not 4<=v<32768 for v in prefix[1:]+chosen[:-1]+rejected[:-1])):
        raise ValueError('Complete native prompt, both full public programs and genuine EOS must fit without clipping')
    result=dict(format=FORMAT,id='semantic-source-contrast:'+row['id'],group_id=row['group_id'],split='train',language='Python',
        prompt=prompt,response=program,rejected_original_public_response=row['response'],
        original_source_function_id=row['id'],donor_source_function_id=donor['id'],
        original_source_group=row['group_id'],donor_source_group=donor['group_id'],
        original_public_URL=row['original_public_URL'],donor_public_URL=donor['original_public_URL'],
        rights=row['rights'],donor_rights=donor['rights'],fixed_cases_sha256=digest(selected),
        original_complete_function_sha256=hashlib.sha256(row['response'].encode()).hexdigest(),
        donor_complete_function_sha256=hashlib.sha256(donor['response'].encode()).hexdigest(),
        actual_first_semantic_failure=dict(position=position,case=case,raw=raw),
        all_original_donor_bytes_kept=True,mechanical_positional_alias_only_when_names_differ=True,
        complete_original_bad_public_implementation_is_negative_only=True,model_generated_targets=False,
        external_neural_teacher=False,human_preferences=False,on_policy_RL=False,
        source_train_diagnostic_not_held_benchmark=True,post_hoc_source_inspection=True,
        finite_numerical_and_original_public_checks_not_general_correctness=True,dataset_admitted_to_training=False)
    pair=dict(id=result['id'],original_source_function_id=row['id'],donor_source_function_id=donor['id'],
        original_source_role='train',donor_source_role='train',native_prompt_ids=prefix,
        chosen_complete_response_EOS_ids=chosen,rejected_complete_response_EOS_ids=rejected,
        positive_program_sha256=hashlib.sha256(program.encode()).hexdigest(),negative_program_sha256=result['original_complete_function_sha256'],
        source_contrast_not_sampled_policy_or_human_preference=True,finite_checks_only=True,optimization_admitted=False)
    return result,prefix+chosen,[-100]*len(prefix)+chosen,pair
