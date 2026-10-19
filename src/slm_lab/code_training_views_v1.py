"""Lossless completion/statement-repair views of verified public train code.

A repair is admitted separately only after its pass-substitution mutant fails
the original published examples. Original function bytes and roles never change.
"""
import ast
import copy
import hashlib

from .code_quality import instruction_code


def sha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()


def original_program(row):
    if (row.get('split')!='train' or row.get('publisher_split')!='train' or row.get('language')!='Python'
            or row.get('parent_source',{}).get('original_partition')!='train'):
        raise ValueError('Only sealed original publisher-train functions can create optimization views')
    program=instruction_code(row['response'])
    if sha(program)!=row['executable_program_sha256'] or sha(row['response'])!=row['unchanged_retained_target_sha256']:
        raise ValueError('Original complete function and formatted target hashes must agree')
    tree=ast.parse(program)
    if len(tree.body)!=1 or type(tree.body[0]) is not ast.FunctionDef:raise ValueError('One complete original function required')
    return program,tree.body[0]


def byte_span(program,node):
    lines=program.encode('utf-8').splitlines(keepends=True)
    start=sum(map(len,lines[:node.lineno-1]))+node.col_offset
    end=sum(map(len,lines[:node.end_lineno-1]))+node.end_col_offset
    value=program.encode('utf-8')[start:end].decode('utf-8')
    if value!=ast.get_source_segment(program,node):raise ValueError('Exact UTF8 source span differs')
    return start,end


def base_row(parent,mode,prompt,response,**metadata):
    return dict(id='code-view:'+mode+':'+parent['parent_source']['id'],group_id=parent['group_id'],split='train',publisher_split='train',
        language='Python',prompt=prompt,response=response,view=mode,parent_training_row_id=parent['id'],
        parent_source=parent['parent_source'],original_source_segment_row=parent['original_source_segment_row'],
        original_complete_program_sha256=parent['executable_program_sha256'],
        original_formatted_target_sha256=parent['unchanged_retained_target_sha256'],
        original_parent_prompt_sha256=sha(parent['prompt']),
        derived_presentation_not_new_unique_function=True,derived_from_model_output=False,external_model_teacher_used=False,
        human_preferences=False,genuine_RLHF_labels=False,dataset_admitted_to_training=False,**metadata)


def completion_view(parent):
    program,function=original_program(parent);body=function.body
    if (isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str)):
        body=body[1:]
    if not body:raise ValueError('Executable complete function body required')
    start,_=byte_span(program,body[0]);raw=program.encode('utf-8');line_start=raw.rfind(b'\n',0,start)+1
    # For ordinary indented bodies, the continuation includes indentation.
    # Inline functions retain their original colon/space in the prefix.
    if raw[line_start:start].strip()==b'':start=line_start
    prefix=raw[:start].decode('utf-8');continuation=raw[start:].decode('utf-8')
    if not prefix or not continuation or prefix+continuation!=program:raise ValueError('Whole exact function reconstruction required')
    prompt=(parent['prompt']+'\n\nContinue the Python function after the provided prefix. Return only the remaining source, then close the code fence.\n```python\n'+prefix)
    response=continuation+'\n```'
    return base_row(parent,'prefix_completion',prompt,response,native_framing='raw_prefix_then_response',
        original_prefix=prefix,original_continuation=continuation,source_split_byte=start,
        reconstructed_program_sha256=sha(prefix+continuation),complete_original_function_bytes_preserved=True,
        full_original_formatted_response_is_this_target=False)


def repair_candidates(parent,maximum=4):
    if type(maximum) is not int or not 1<=maximum<=16:raise ValueError('Explicit bounded statement-search budget required')
    program,function=original_program(parent);raw=program.encode('utf-8')
    priority={ast.Return:0,ast.Raise:1,ast.Assign:2,ast.AugAssign:3,ast.If:4,ast.For:5,ast.While:6,ast.Expr:7}
    nodes=[n for n in ast.walk(function) if type(n) in priority
        and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
    nodes.sort(key=lambda n:(priority[type(n)],n.lineno,n.col_offset,n.end_lineno,n.end_col_offset))
    marker='__CODE_SLOT_'+parent['parent_source']['id'][:12]+'__'
    if marker in program or marker in parent['prompt']:raise ValueError('Unambiguous derived placeholder required')
    output=[]
    for node in nodes:
        start,end=byte_span(program,node);prefix=raw[:start].decode('utf-8');middle=raw[start:end].decode('utf-8');suffix=raw[end:].decode('utf-8')
        mutant=prefix+'pass'+suffix
        try:ast.parse(mutant)
        except SyntaxError:continue
        if mutant==program:continue
        if prefix+middle+suffix!=program:raise ValueError('Every full original program byte must reconstruct')
        marked=prefix+marker+suffix
        prompt=('Repair the marked statement in this Python function. Replace the marker with the original working statement. '
            'Return only the replacement source, preserving indentation inside a multiline statement.\n\n'
            'Original source documentation and declared context:\n'+parent['prompt']+
            '\n\nFunction with one statement removed:\n```python\n'+marked+'\n```\n\nReplacement statement:')
        row=base_row(parent,'verified_statement_repair',prompt,middle,native_framing='user_assistant',
            original_prefix=prefix,original_suffix=suffix,original_middle=middle,placeholder=marker,
            source_span_bytes=[start,end],source_span_lines=[node.lineno,node.col_offset,node.end_lineno,node.end_col_offset],
            statement_type=type(node).__name__,mutant_program=mutant,mutant_program_sha256=sha(mutant),
            reconstructed_program_sha256=sha(prefix+middle+suffix),complete_original_function_bytes_preserved=True,
            mutant_is_machine_derived_not_publisher_code=True,full_original_formatted_response_is_this_target=False,
            published_example_failure_gate_pending=True)
        output.append(row)
        if len(output)==maximum:break
    return output


def encode_view(row,tokenizer,maximum=4096):
    if [tokenizer.token_to_id(t) for t in ['<bos>','<eos>']]!=[1,2]:raise ValueError('Exact own tokenizer framing required')
    if row['native_framing']=='raw_prefix_then_response':prefix=row['prompt']
    elif row['native_framing']=='user_assistant':prefix='User: '+row['prompt']+'\n\nAssistant:'
    else:raise ValueError('Known complete view framing required')
    before=[1,*tokenizer.encode(prefix).ids];target=[*tokenizer.encode(row['response']).ids,2]
    if not row['response'] or len(target)<2 or len(before)+len(target)>maximum:
        raise ValueError('Complete source/input/target must fit; no truncation allowed')
    if tokenizer.decode(before[1:])!=prefix or tokenizer.decode(target[:-1])!=row['response']:
        raise ValueError('Lossless complete native text/target tokenization required')
    return before+target,[-100]*len(before)+target


def verify_mutation(candidate,reference_observation,mutant,*,seconds=5):
    """Keep the original publisher record separate from machine-derived code."""
    from .documented_doctest_v1 import execute
    if (candidate['source'].get('original_partition')!='train' or candidate['source'].get('language')!='Python'
            or reference_observation.get('source_id')!=candidate['source']['id']
            or reference_observation.get('program_sha256')!=candidate['program_sha256']
            or reference_observation.get('all_original_examples_passed') is not True):
        raise ValueError('Exact independently verified original train source required')
    scored=[i for i,r in enumerate(reference_observation['raw_results']) if r['entry_body_calls']>0]
    if not scored or mutant==candidate['program']:raise ValueError('A real change and exercised source examples required')
    derived=copy.deepcopy(candidate);formatted='```python\n'+mutant+'\n```'
    derived['source']=dict(candidate['source'],id='machine-pass-mutation:'+candidate['source']['id'],response=formatted,
        annotation_origin='Deterministic pass-substitution mutation; not publisher code, own-policy output or a human label.')
    derived.update(program=mutant,program_sha256=sha(mutant),existing_formatted_source_target_sha256=sha(formatted))
    receipt=dict(original_source_id=candidate['source']['id'],original_program_sha256=candidate['program_sha256'],
        mutant_program_sha256=sha(mutant),mutation_origin='deterministic_pass_substitution',
        scored_original_example_indices=scored,reference_outputs_stay_outside_worker=True,
        original_publisher_record_not_overwritten=True,human_preference_label=False,own_policy_generation=False)
    try:actual=execute(derived,seconds=seconds)
    except (ValueError,SyntaxError) as e:
        return dict(receipt,qualifies_as_observed_failure=False,status='static_contract_qualification',qualification=str(e))
    rows=actual.get('raw_results',[]);flags=actual.get('example_passes',[])
    observed=actual.get('all_original_examples_observed') is True and len(rows)==candidate['whole_original_example_count'] and len(flags)==len(rows)
    passed=sum(bool(flags[i]) and rows[i]['entry_body_calls']>0 for i in scored) if observed else None
    reward=passed/len(scored) if observed else None
    return dict(receipt,status=actual['status'],all_original_examples_observed=observed,
        passed_scored_original_examples=passed,scored_original_examples=len(scored),fraction_passed=reward,
        qualifies_as_observed_failure=observed and reward<1.,raw_machine_mutant_execution=actual)
