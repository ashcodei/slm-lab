"""Whole public direct targets and contract-compatible correction contrasts."""
import ast
import copy
import hashlib

from .public_function_contract_census_v2 import cases,digest,family,identity,passes


def source_only(row):
    if (family(row) is None or row['response']!=row['source_candidate']['original_function_program']
            or row['id']!=row['source_candidate']['id']):raise ValueError('Exact original reviewed train source required')


def encode(prompt,program,tokenizer):
    prefix=[1]+tokenizer.encode('User: '+prompt+'\nAssistant:',add_special_tokens=False).ids
    target=tokenizer.encode(program,add_special_tokens=False).ids+[2]
    if (len(prefix)+len(target)>4096 or len(target)<2
            or tokenizer.decode(prefix[1:],skip_special_tokens=False)!='User: '+prompt+'\nAssistant:'
            or tokenizer.decode(target[:-1],skip_special_tokens=False)!=program
            or any(not 4<=v<32768 for v in prefix[1:]+target[:-1])):
        raise ValueError('Both complete public programs, exact prompt bytes and true EOS must fit without truncation')
    return prefix+target,[-100]*len(prefix)+target


def direct(row,tokenizer):
    source_only(row);candidate=copy.deepcopy(row)
    candidate.update(id='contract-qualified-public:'+row['id'],original_source_function_id=row['id'],
        contract_family=family(row),source_function_identity=identity(row),independent_contract_cases_sha256=digest(cases(row)),
        original_prompt_and_whole_response_unchanged=True,finite_qualification_not_general_correctness=True,
        source_train_not_independent_benchmark=True,dataset_admitted_to_training=False)
    ids,labels=encode(row['prompt'],row['response'],tokenizer)
    return candidate,ids,labels


def alternative(row,donor):
    source_only(row);source_only(donor)
    if family(row)!=family(donor) or row['id']==donor['id']:raise ValueError('Two distinct reviewed implementations of the same exact contract required')
    target=ast.parse(row['response']).body;other=ast.parse(donor['response']).body
    if (len(target)!=1 or len(other)!=1 or not isinstance(target[0],ast.FunctionDef) or not isinstance(other[0],ast.FunctionDef)
            or len(target[0].args.args)!=len(other[0].args.args) or target[0].args.posonlyargs or other[0].args.posonlyargs
            or target[0].args.defaults or other[0].args.defaults):
        raise ValueError('Exact compatible complete positional public interfaces required')
    target=target[0];other=other[0]
    if target.name==other.name:return donor['response']
    names=[arg.arg for arg in target.args.args];doc=ast.get_docstring(target,clean=False)
    alias='def '+target.name+'('+', '.join(names)+'):\n'
    if doc is not None:alias+='    '+repr(doc)+'\n'
    alias+='    return '+other.name+'('+', '.join(names)+')\n'
    return donor['response']+'\n\n'+alias


def contrast(row,donor,negative,tokenizer):
    selected=cases(row);program=alternative(row,donor)
    if (negative.get('source_id')!=row['id'] or negative.get('program_sha256')!=hashlib.sha256(row['response'].encode()).hexdigest()
            or negative.get('cases_sha256')!=digest(selected) or negative.get('all_cases_observed') is not True
            or len(negative.get('raw_results',[]))!=len(selected)):
        raise ValueError('Every actual original program and explicit contract observation must reconstruct')
    checks=[passes(c,r) for c,r in zip(selected,negative['raw_results'])]
    if checks!=negative.get('case_passes') or all(checks) or negative.get('all_cases_passed') is not False:
        raise ValueError('A complete actual contract counterexample is required')
    position=next(i for i,v in enumerate(checks) if not v);case=selected[position];observed=negative['raw_results'][position]
    prompt=('Repair this original public Python implementation to satisfy its documented contract. Return the complete implementation and any needed helpers.\n\n'+
        row['prompt']+'\n\nOriginal implementation with a verified counterexample:\n```python\n'+row['response']+
        '\n```\n\nCounterexample arguments: '+repr(case['args'])+'\nExpected return: '+repr(case['expected'])+
        ('\nExpected arguments after call: '+repr(case['expected_post_call_args']) if 'expected_post_call_args' in case else '')+
        '\nObserved return and arguments after call: '+repr(observed['stdout'])+'\nObserved exception: '+repr(observed['exception_only'])+'\n')
    ids,labels=encode(prompt,program,tokenizer);negative_ids,negative_labels=encode(prompt,row['response'],tokenizer)
    start=next(i for i,v in enumerate(labels) if v!=-100);negative_start=next(i for i,v in enumerate(negative_labels) if v!=-100)
    if start!=negative_start or ids[:start]!=negative_ids[:start]:raise ValueError('Matched complete positive and negative prompt masks required')
    key=digest([row['id'],donor['id']]);candidate=dict(id='contract-source-contrast:'+key,group_id=row['group_id'],split='train',language='Python',
        prompt=prompt,response=program,rejected_original_public_response=row['response'],contract_family=family(row),
        original_source_function_id=row['id'],donor_source_function_id=donor['id'],original_source_group=row['group_id'],donor_source_group=donor['group_id'],
        original_public_URL=row['original_public_URL'],donor_public_URL=donor['original_public_URL'],rights=row['rights'],donor_rights=donor['rights'],
        original_complete_source_sha256=hashlib.sha256(row['response'].encode()).hexdigest(),
        donor_complete_source_sha256=hashlib.sha256(donor['response'].encode()).hexdigest(),counterexample=dict(case=case,raw=observed),
        source_and_donor_roles='train',all_original_donor_bytes_unchanged=True,only_static_positional_alias_added=True,
        original_bad_source_used_only_as_negative=True,source_train_not_independent_benchmark=True,human_preferences=False,on_policy_RL=False,
        external_neural_teacher=False,model_generated_targets=False,finite_qualification_not_general_correctness=True,dataset_admitted_to_training=False)
    pair=dict(id=candidate['id'],original_source_function_id=row['id'],donor_source_function_id=donor['id'],
        contract_family=family(row),original_source_role='train',donor_source_role='train',native_prompt_ids=ids[:start],
        chosen_complete_response_EOS_ids=ids[start:],rejected_complete_response_EOS_ids=negative_ids[start:],
        finite_qualified_machine_source_contrast_not_human_or_on_policy_RL=True,optimization_admitted=False)
    return candidate,ids,labels,pair
