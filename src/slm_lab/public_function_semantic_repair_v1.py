"""Source-faithful semantic bug/repair candidates, verified by public examples.

One AST operator or literal changes inside the original function body. The
correct target stays the complete original public implementation. Mutations
are programmatic source derivatives, never sampled model trajectories, human
judgments, proofs of general correctness, or independent benchmark questions.
"""
import ast
from copy import deepcopy
import doctest
import hashlib
import json

from .own_function_result_readback_v1 import effect

FORMAT = 'public-original-function-semantic-counterexample-repair-v1'
OPERATORS = {ast.Add: (ast.Sub,), ast.Sub: (ast.Add,), ast.Mult: (ast.Add,),
    ast.Div: (ast.Mult,), ast.FloorDiv: (ast.Div,), ast.Mod: (ast.FloorDiv,), ast.Pow: (ast.Mult,),
    ast.BitAnd: (ast.BitOr,), ast.BitOr: (ast.BitAnd,), ast.BitXor: (ast.BitOr,),
    ast.LShift: (ast.RShift,), ast.RShift: (ast.LShift,), ast.And: (ast.Or,), ast.Or: (ast.And,),
    ast.Eq: (ast.NotEq,), ast.NotEq: (ast.Eq,), ast.Lt: (ast.LtE,ast.Gt),
    ast.LtE: (ast.Lt,ast.GtE), ast.Gt: (ast.GtE,ast.Lt), ast.GtE: (ast.Gt,ast.LtE),
    ast.USub: (ast.UAdd,), ast.UAdd: (ast.USub,)}


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def source_only(row):
    source=row['source_candidate']
    if (row['split']!='train' or source['role']!='train' or source['id']!=row['id']
            or source['original_function_program']!=row['response']):
        raise ValueError('Only the complete admitted original source-train function is eligible')
    return source


def walk(node,path):
    yield path,node
    for field,value in ast.iter_fields(node):
        if isinstance(value,ast.AST):yield from walk(value,path+[field])
        elif isinstance(value,list):
            for i,child in enumerate(value):
                if isinstance(child,ast.AST):yield from walk(child,path+[field,i])


def at(tree,path):
    for key in path:tree=tree[key] if isinstance(key,int) else getattr(tree,key)
    return tree


def candidates(row,*,maximum=16):
    source=source_only(row)
    if type(maximum) is not int or not 1<=maximum<=64:raise ValueError('Explicit bounded mutation count required')
    original=ast.parse(row['response'])
    matches=[(i,node) for i,node in enumerate(original.body) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==source['function']]
    if len(matches)!=1:raise ValueError('Exactly one original public function entrypoint required')
    index,function=matches[0];result=[];seen=set()
    for i,statement in enumerate(function.body):
        for path,node in walk(statement,['body',index,'body',i]):
            replacements=[]
            if type(node) in OPERATORS:replacements=[(kind(),kind.__name__) for kind in OPERATORS[type(node)]]
            elif isinstance(node,ast.Constant) and type(node.value) is int and abs(node.value)<=1000000:
                replacements=[(ast.copy_location(ast.Constant(value=node.value+1),node),'integer_plus_one')]
            elif isinstance(node,ast.Constant) and type(node.value) is bool:
                replacements=[(ast.copy_location(ast.Constant(value=not node.value),node),'boolean_flip')]
            for replacement,kind in replacements:
                changed=deepcopy(original);parent=at(changed,path[:-1]);last=path[-1]
                if isinstance(last,int):parent[last]=replacement
                else:setattr(parent,last,replacement)
                changed=ast.fix_missing_locations(changed);program=ast.unparse(changed)+'\n';digest=sha(program)
                if digest in seen:continue
                seen.add(digest)
                rebuilt=ast.parse(program);new_function=rebuilt.body[index]
                if (ast.dump(new_function.args)!=ast.dump(function.args)
                        or ast.get_docstring(new_function,clean=False)!=ast.get_docstring(function,clean=False)
                        or new_function.name!=function.name):raise ValueError('Mutation must preserve signature and every original literal public example')
                result.append(dict(mutation_position=len(result),kind=kind,path=path,
                    original_node_ast=ast.dump(node),replacement_node_ast=ast.dump(replacement),
                    original_complete_program_sha256=sha(row['response']),mutant_program_sha256=digest,program=program))
                if len(result)==maximum:return result
    return result


def compact(observed):
    actual=effect(observed)
    return {**{k:v for k,v in actual.items() if k not in {'raw_results','example_passes'}},
        'whole_public_execution_effect_sha256':sha(json.dumps(actual,sort_keys=True,separators=(',',':')))}


def counterexample(row,observed):
    source=source_only(row)
    if (observed.get('status')!='generated_public_example_failure' or not observed.get('syntax_valid')
            or not observed.get('all_original_examples_observed') or observed.get('all_original_examples_passed')
            or observed.get('entry_body_calls',0)<1):return None
    tests=doctest.DocTestParser().get_examples(source['original_literal_docstring'])
    if len(tests)!=source['whole_original_example_count'] or len(observed['example_passes'])!=len(tests):
        raise ValueError('Every original public example must be retained before feedback selection')
    index=next(i for i,passed in enumerate(observed['example_passes']) if not passed)
    raw=observed['raw_results'][index]
    result=dict(first_failing_original_public_example_index=index,public_example_input=tests[index].source,
        published_expected_output=tests[index].want,actual_stdout=raw['stdout'],actual_exception=raw['exception_only'],
        original_public_examples=len(tests),all_original_public_examples_observed=True)
    if len(json.dumps(result).encode())>16384:return None
    return result


def view(row,mutation,feedback,tokenizer):
    source=source_only(row)
    if mutation['original_complete_program_sha256']!=sha(row['response']) or mutation['mutant_program_sha256']!=sha(mutation['program']):
        raise ValueError('Exact original and single-mutation program bytes required')
    prompt=('Repair this Python function according to its original public examples. Keep the function name and parameters. '
        'Return the complete corrected function only. The declared standard-library context is available.\n\n'
        'Declared standard-library context:\n'+source['declared_prelude']+'\n\nCandidate implementation:\n```python\n'+mutation['program']+'```\n\n'
        'Observed feedback from an original public example:\n'+json.dumps(feedback,ensure_ascii=False,sort_keys=True)+'\n')
    prefix=[1,*tokenizer.encode('User: '+prompt+'\nAssistant:',add_special_tokens=False).ids]
    target=[*tokenizer.encode(row['response'],add_special_tokens=False).ids,2];ids=prefix+target
    if (tokenizer.decode(prefix[1:],skip_special_tokens=False)!='User: '+prompt+'\nAssistant:'
            or tokenizer.decode(target[:-1],skip_special_tokens=False)!=row['response']
            or any(not 4<=token<32768 for token in ids[1:-1])):
        raise ValueError('Exact untruncated own-tokenizer source bytes and special-token boundaries required')
    if len(ids)>4096:return None,dict(disposition='complete_native_context_exceeds4096',context_tokens=len(ids),target_tokens=len(target))
    result=dict(id='semantic-repair:'+row['id'],group_id=row['group_id'],split='train',language='Python',
        source_function_id=row['id'],prompt=prompt,response=row['response'],original_public_URL=row['original_public_URL'],rights=row['rights'],
        original_literal_docstring_sha256=sha(source['original_literal_docstring']),mutation=mutation,public_counterexample=feedback,
        complete_original_function_target_unchanged=True,programmatic_mutation_not_a_model_rollout=True,
        public_example_answers_visible_in_model_prompt=True,human_preferences=False,external_neural_teacher=False,
        finite_published_checks_not_general_correctness_or_efficiency=True,dataset_admitted_to_training=False)
    return (result,ids,[-100]*len(prefix)+target),dict(disposition='complete_semantic_repair_candidate',context_tokens=len(ids),target_tokens=len(target))
