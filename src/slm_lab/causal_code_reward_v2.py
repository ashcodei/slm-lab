"""Published-stream rewards that include post-call value/side-effect checks.

Reference no-op ablation identifies observable checks that depend on the entry
body. Assignment-only setup cannot earn reward. Full reward additionally needs
every original example to pass. This is finite machine feedback, not RLHF.
"""
import ast
import copy

from .code_quality import instruction_code
from .code_training_views_v1 import byte_span,sha,verify_mutation
from .documented_doctest_v1 import contract,compare,execute


def pass_ablation(candidate):
    program=candidate['program'];tree=ast.parse(program)
    if len(tree.body)!=1 or type(tree.body[0]) is not ast.FunctionDef:raise ValueError('One complete entry function required')
    body=tree.body[0].body
    if isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):body=body[1:]
    if not body:raise ValueError('No executable source body for a behavior ablation')
    start,_=byte_span(program,body[0]);_,end=byte_span(program,body[-1]);raw=program.encode('utf-8')
    return raw[:start].decode('utf-8')+'pass'+raw[end:].decode('utf-8')


def ablate(candidate,reference,seconds=5):
    mutated=pass_ablation(candidate)
    if mutated==candidate['program']:
        return dict(source_id=candidate['source']['id'],supported=False,reason='Original executable body is already a no-op',
                    original_program_sha256=candidate['program_sha256'])
    result=verify_mutation(candidate,reference,mutated,seconds=seconds);observation=result.get('raw_machine_mutant_execution')
    if not observation or observation.get('all_original_examples_observed') is not True:
        return dict(source_id=candidate['source']['id'],supported=False,reason='Complete bounded no-op counterfactual was not observed',
                    original_program_sha256=candidate['program_sha256'],counterfactual=result)
    examples=contract(candidate);flags=[compare(example,row) for example,row in zip(examples,observation['raw_results'])]
    if len(flags)!=len(examples) or flags!=observation['example_passes']:raise ValueError('Every actual counterfactual output must reconstruct')
    indices=[i for i,passed in enumerate(flags) if not passed]
    return dict(source_id=candidate['source']['id'],supported=bool(indices),
        reason='Observed entry-body-dependent published checks' if indices else 'Published examples do not distinguish the no-op body',
        original_program_sha256=candidate['program_sha256'],ablation_program_sha256=sha(mutated),
        dependent_example_indices=indices,complete_original_example_count=len(examples),
        original_entry_body_call_indices=[i for i,r in enumerate(reference['raw_results']) if r['entry_body_calls']>0],
        post_call_dependent_check_indices=[i for i in indices if reference['raw_results'][i]['entry_body_calls']==0],
        counterfactual=result,own_policy_generation=False,human_preference_label=False)


def evaluate(candidate,reference,ablation,response,*,origin='own_policy',seconds=5):
    if (candidate['source'].get('original_partition')!='train' or candidate['source'].get('language')!='Python'
            or reference.get('source_id')!=candidate['source']['id'] or reference.get('program_sha256')!=candidate['program_sha256']
            or reference.get('all_original_examples_passed') is not True
            or ablation.get('source_id')!=candidate['source']['id'] or ablation.get('original_program_sha256')!=candidate['program_sha256']
            or origin not in ('own_policy','test_fixture')):
        raise ValueError('Exact original train source, complete reference and declared response origin required')
    receipt=dict(format='own-causal-published-stream-code-reward-v2',original_source_id=candidate['source']['id'],
        original_program_sha256=candidate['program_sha256'],candidate_origin=origin,
        supported=ablation['supported'],reward=0.,full_original_stream_passed=None,
        causal_example_indices=ablation.get('dependent_example_indices',[]),
        expected_outputs_stay_outside_worker=True,original_implementation_not_sent_to_candidate_worker=True,
        pure_setup_is_not_scored=True,full_reward_requires_every_published_example=True,
        partial_reward_ceiling=.9,finite_published_examples_not_general_correctness_or_efficiency=True,
        human_preference_label=False,external_model_teacher=False,goal_achieved=False)
    if not ablation['supported']:return dict(receipt,status='source_reward_uninformative_use_public_target_CE',not_a_candidate_correctness_verdict=True)
    expected=pass_ablation(candidate)
    if ablation.get('ablation_program_sha256')!=sha(expected):raise ValueError('Exact no-op counterfactual identity required')
    examples=contract(candidate);control=ablation['counterfactual']['raw_machine_mutant_execution']
    flags=[compare(example,row) for example,row in zip(examples,control['raw_results'])]
    if len(flags)!=len(examples) or ablation['dependent_example_indices']!=[i for i,x in enumerate(flags) if not x]:
        raise ValueError('Every scored behavior-dependent original example must reconstruct')
    if not isinstance(response,str):raise ValueError('Actual response text required')
    code=instruction_code(response);derived=copy.deepcopy(candidate);formatted='```python\n'+code+'\n```'
    derived['source']=dict(candidate['source'],id='candidate:'+origin+':'+candidate['source']['id'],response=formatted,
        annotation_origin='Own policy output' if origin=='own_policy' else 'Mechanics-test fixture; not model output')
    derived.update(program=code,program_sha256=sha(code),existing_formatted_source_target_sha256=sha(formatted))
    receipt.update(candidate_response_sha256=sha(response),candidate_program_sha256=sha(code))
    try:observation=execute(derived,seconds=seconds)
    except (ValueError,SyntaxError) as error:return dict(receipt,status='candidate_contract_or_context_qualification',qualification=str(error))
    rows=observation.get('raw_results',[])
    if observation.get('all_original_examples_observed') is not True or len(rows)!=len(examples):
        return dict(receipt,status='incomplete_candidate_execution',raw_candidate_execution=observation)
    passed=[compare(example,row) for example,row in zip(examples,rows)]
    if passed!=observation['example_passes']:raise ValueError('Actual candidate raw outputs and checker decisions differ')
    body_calls=sum(r['entry_body_calls'] for r in rows);full=all(passed);indices=ablation['dependent_example_indices']
    partial=sum(passed[i] for i in indices)/len(indices)
    reward=(1. if full else .9*partial) if body_calls>0 else 0.
    return dict(receipt,reward=reward,full_original_stream_passed=full,scored_examples_passed=sum(passed[i] for i in indices),
        scored_examples=len(indices),actual_entry_body_calls=body_calls,status='complete_published_stream_scored',
        raw_candidate_execution=observation)
