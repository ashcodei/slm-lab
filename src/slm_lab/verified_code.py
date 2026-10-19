"""Public training programs checked against published literal examples.

Verification uses the constrained Python worker, not unrestricted execution. Passing a
small set of source examples does not prove general correctness or efficiency.
"""
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import doctest
import hashlib
import json
import math
from pathlib import Path

from tokenizers import Tokenizer

from .code_quality import execute, instruction_code
from .code_tasks import TASKS
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .wikipedia_corpus import protected_questions


def canonical_code(code):
    tree=ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.body:
            if isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
                node.body.pop(0)
                if not node.body:node.body.append(ast.Pass())
    return ast.unparse(tree)


def literal(value):
    def validate(item,depth=0):
        if depth>12:raise ValueError('Example nesting too deep')
        if type(item) in {int,float,str,bool,type(None)}:
            if isinstance(item,float) and not math.isfinite(item):raise ValueError('Non-finite example')
            return
        if isinstance(item,(list,tuple)):
            for child in item:validate(child,depth+1)
            return
        if isinstance(item,dict):
            for key,child in item.items():validate(key,depth+1);validate(child,depth+1)
            return
        raise ValueError('Unsupported example value')
    validate(value);encoded=repr(value)
    if len(encoded)>100000:raise ValueError('Example exceeds size budget')
    return encoded


def call_case(call,expected):
    if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.keywords:
        raise ValueError('Example must call a named function with literal positional arguments')
    args=tuple(ast.literal_eval(node) for node in call.args)
    return call.func.id,{'args_literal':literal(args),'expected_literal':literal(expected)}


def assertion_cases(tests):
    cases=[];name=None
    for text in tests:
        statements=ast.parse(text).body
        if len(statements)!=1 or not isinstance(statements[0],ast.Assert):raise ValueError('Expected a single assertion')
        expression=statements[0].test
        if isinstance(expression,ast.Compare) and len(expression.ops)==1 and isinstance(expression.ops[0],ast.Eq):
            current,case=call_case(expression.left,ast.literal_eval(expression.comparators[0]))
        elif isinstance(expression,ast.Call):current,case=call_case(expression,True)
        elif isinstance(expression,ast.UnaryOp) and isinstance(expression.op,ast.Not):current,case=call_case(expression.operand,False)
        else:raise ValueError('Unsupported assertion structure')
        if name is not None and name!=current:raise ValueError('Examples call different entry points')
        name=current;cases.append(case)
    return name,cases


def doctest_cases(description,name):
    cases=[]
    try:examples=doctest.DocTestParser().get_examples(description)
    except ValueError:return []
    for example in examples:
        try:
            statement=ast.parse(example.source).body
            if len(statement)!=1 or not isinstance(statement[0],ast.Expr):continue
            current,case=call_case(statement[0].value,ast.literal_eval(example.want.strip()))
            if current==name:cases.append(case)
        except (SyntaxError,ValueError,TypeError):continue
    return cases


def renamed(code,name,replacement):
    class Rename(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id==name:node.id=replacement
            return node
        def visit_FunctionDef(self,node):
            if node.name==name:node.name=replacement
            return self.generic_visit(node)
    return ast.unparse(Rename().visit(ast.parse(canonical_code(code))))


def check(program):
    payload={'language':'Python','code':program['code'],'function':program['function'],
             'cases':program['cases'],'stress':[]}
    return execute(payload)


def prepare(output,source_cache,functions,tokenizer,questions,variants=4,documented_functions=None,existing_verified=None):
    if not 1<=variants<=8:raise ValueError('Use one to eight explicitly recorded name variants')
    output,source_cache,functions,tokenizer,questions=map(Path,[output,source_cache,functions,tokenizer,questions])
    output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists():raise ValueError('Use a new output for a changed verification run')
    source_meta=json.loads((source_cache/'sources.json').read_text())
    for item in source_meta['files']:
        if file_hash(source_cache/item['path'])!=item['sha256']:raise ValueError('Pinned MBPP source changed')
    notices=(source_cache/'repository-README.md').read_text()
    if 'All datasets in this repository are released under the CC BY 4.0' not in notices:
        raise ValueError('Publisher dataset license notice missing')
    tok=Tokenizer.from_file(str(tokenizer));matcher=protected_questions(questions)
    existing_rows=[];existing_evidence=[];additional_inputs={}
    if existing_verified:
        existing_verified=Path(existing_verified);existing_manifest=json.loads((existing_verified/'manifest.json').read_text())
        if existing_manifest['requested_name_variants_per_program']!=variants:raise ValueError('Existing verified name-variant budget differs')
        for key,path in [('questions_sha256',questions),('tokenizer_sha256',tokenizer)]:
            if existing_manifest['inputs'][key]!=file_hash(path):raise ValueError('Existing verified tokenizer/questions differ')
        if existing_manifest['inputs']['mbpp_sources']!=source_meta or existing_manifest['inputs']['functions_sha256']!=file_hash(functions):
            raise ValueError('Existing verified public source inputs differ')
        for name,digest in existing_manifest['files_sha256'].items():
            if file_hash(existing_verified/name)!=digest:raise ValueError('Previously verified records changed')
        existing_rows=list(jsonl(existing_verified/'train.jsonl'));existing_evidence=list(jsonl(existing_verified/'verification.jsonl'))
        additional_inputs['existing_verified_manifest_sha256']=file_hash(existing_verified/'manifest.json')
    all_mbpp=list(jsonl(source_cache/'mbpp.jsonl'))
    reserved_codes=set();reserved_prompts=set()
    for row in all_mbpp:
        if row['task_id']<=600:
            reserved_prompts.add(' '.join(row['text'].casefold().split()))
            try:reserved_codes.add(hashlib.sha256(canonical_code(row['code']).encode()).hexdigest())
            except SyntaxError:pass
    # Also keep the existing local evaluation specifications and references out
    # of this supplemental training source selection.
    reserved_prompts.update(' '.join(task[2].casefold().split()) for task in TASKS)
    reserved_codes.update(hashlib.sha256(canonical_code(task[4]).encode()).hexdigest() for task in TASKS)
    proposals=[];excluded=[];seen={row['original_program_sha256'] for row in existing_rows}

    def propose(code,name,cases,description,origin):
        try:
            normalized=canonical_code(code);digest=hashlib.sha256(normalized.encode()).hexdigest()
            if digest in seen:raise ValueError('Duplicate normalized program')
            if digest in reserved_codes:raise ValueError('Exact reserved reference program')
            normalized_prompt=' '.join(description.casefold().split())
            if any(text in normalized_prompt for text in reserved_prompts if text):raise ValueError('Reserved benchmark specification')
            if list(matcher.iter(normalized_prompt)):raise ValueError('Reserved knowledge question')
            if len(cases)<2:raise ValueError('Fewer than two parseable source examples')
            if name not in {node.name for node in ast.parse(normalized).body if isinstance(node,ast.FunctionDef)}:
                raise ValueError('Entry point is not a top-level function')
            seen.add(digest)
            proposals.append({'code':normalized,'function':name,'cases':cases,'description':description,
                              'origin':origin,'code_sha256':digest})
        except (ValueError,SyntaxError) as error:
            excluded.append({'source_id':origin['id'],'reason':str(error)})

    for row in all_mbpp:
        if not 601<=row['task_id']<=974:continue
        if existing_verified:continue
        origin={'id':'mbpp:'+str(row['task_id']),'group_id':'mbpp:'+str(row['task_id']),
                'source':'Google Research / MBPP official training split','license':'CC-BY-4.0',
                'license_url':'https://creativecommons.org/licenses/by/4.0/',
                'source_url':next(item['url'] for item in source_meta['files'] if item['path']=='mbpp.jsonl'),'revision':source_meta['revision']}
        try:
            name,cases=assertion_cases(row['test_list']+row.get('challenge_test_list',[]))
            propose(row.get('test_setup_code','')+'\n'+row['code'],name,cases,row['text'],origin)
        except (ValueError,SyntaxError,TypeError) as error:excluded.append({'source_id':origin['id'],'reason':str(error)})
    for row in jsonl(functions):
        if existing_verified:break
        origin={key:row[key] for key in ['id','group_id','source','source_url','license','revision']}
        if row.get('license_url'):origin['license_url']=row['license_url']
        cases=doctest_cases(row['prompt'],row['function'])
        if cases:propose(instruction_code(row['response']),row['function'],cases,row['prompt'],origin)
    if documented_functions:
        documented_functions=Path(documented_functions)
        corpus_path=documented_functions.parent/'corpus.json'
        corpus_manifest=json.loads(corpus_path.read_text())
        if file_hash(documented_functions)!=corpus_manifest['sft_files_sha256']['train']:
            raise ValueError('Additional documented examples are not the pinned training file')
        additional_inputs['documented_functions_sha256']=file_hash(documented_functions)
        additional_inputs['documented_corpus_manifest_sha256']=file_hash(corpus_path)
        for row in jsonl(documented_functions):
            if row.get('language','').casefold()!='python' or '>>>' not in row['prompt']:continue
            code=instruction_code(row['response'])
            try:
                definitions=[node for node in ast.parse(code).body if isinstance(node,ast.FunctionDef)]
                if not definitions:continue
                name=definitions[0].name;cases=doctest_cases(row['prompt'],name)
            except SyntaxError:continue
            if len(cases)<2:
                excluded.append({'source_id':row['id'],'reason':'Fewer than two parseable literal published examples; malformed/interactive examples are not repaired'})
                continue
            origin={key:value for key,value in row.items() if key not in {'prompt','response','annotation_origin','environment'}}
            origin['original_environment']=row.get('environment')
            origin['original_annotation_origin']=row.get('annotation_origin')
            origin['source_dataset_manifest']=str(corpus_path.resolve())
            origin['revision']='Original GitHub revision in source_url; CodeSearchNet archive pinned by '+file_hash(corpus_path)
            propose(code,name,cases,row['prompt'],origin)
    accepted=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for program,result in zip(proposals,pool.map(check,proposals)):
            if not result.get('functional_passed'):
                excluded.append({'source_id':program['origin']['id'],'reason':'Published examples failed or exceed the supported runner subset','worker':result})
                continue
            accepted.append(program)
            if len(accepted)%50==0:
                atomic_json(output/'progress.json',{'status':'checking public training examples','accepted_programs':len(accepted),'proposals':len(proposals)})
    # Name variants preserve source behavior; every generated variant is checked
    # again with the same published examples. They add exposures, not algorithms.
    rows=existing_rows.copy();verified=existing_evidence.copy()
    for program in accepted:
        for index in range(variants):
            name=program['function'] if index==0 else 'solve_'+hashlib.sha256((program['origin']['id']+str(index)).encode()).hexdigest()[:12]
            code=renamed(program['code'],program['function'],name)
            transformed={**program,'code':code,'function':name};result=check(transformed)
            if not result.get('functional_passed'):raise ValueError('Behavior-preserving name variant failed: '+program['origin']['id'])
            function=next(node for node in ast.parse(code).body if isinstance(node,ast.FunctionDef) and node.name==name)
            description=program['description'].replace(program['function']+'(',name+'(')
            signature='def '+name+'('+ast.unparse(function.args)+'):'
            examples='\n'.join(f"{name}{case['args_literal']} -> {case['expected_literal']}" for case in program['cases'][:4])
            prompt=f'Implement this Python function:\n\n{signature}\n\n{description}\n\nPublished behavior examples:\n{examples}\nReturn the complete function and any necessary helpers.'
            response='```python\n'+code+'\n```'
            if len(tok.encode(response).ids)>2047 or len(tok.encode(prompt).ids)+len(tok.encode(response).ids)+16>4096:
                excluded.append({'source_id':program['origin']['id'],'reason':'Verified variant exceeds the current token budget'});continue
            identifier=hashlib.sha256((program['origin']['id']+str(index)).encode()).hexdigest()
            row={**program['origin'],'id':'verified-'+identifier,'language':'Python','function':name,
                 'prompt':prompt,'response':response,'original_program_sha256':program['code_sha256'],
                 'variant_index':index,'verification_cases':program['cases'],
                 'verification':'Passed original published literal examples in the constrained worker; not proof of general correctness/efficiency',
                 'annotation_origin':'Public source description/implementation/examples with static name/format transformations; authorship is not independently verified; not human preference feedback.'}
            rows.append(row)
        verified.append({'source_id':program['origin']['id'],'program_sha256':program['code_sha256'],'cases':len(program['cases'])})
    if not rows:raise ValueError('No verified training programs retained')
    (output/'train.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=True)+'\n' for row in rows))
    (output/'verification.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=True)+'\n' for row in verified))
    (output/'excluded.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=True)+'\n' for row in excluded))
    reserved=[row for row in all_mbpp if row['task_id']<=600]
    (output/'reserved-mbpp.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=True)+'\n' for row in reserved))
    manifest={'created_by':'Published-example verification and static name transformation',
              'train_examples':len(rows),'unique_programs':len({row['original_program_sha256'] for row in rows}),
              'programs_passing_original_examples':len(verified),'requested_name_variants_per_program':variants,
              'by_source':dict(Counter(row['source'] for row in rows)),
              'official_mbpp_training_ids':[601,974],'reserved_mbpp_ids':[1,600],
              'inputs':{'mbpp_sources':source_meta,'functions_sha256':file_hash(functions),
                        'questions_sha256':file_hash(questions),'tokenizer_sha256':file_hash(tokenizer),**additional_inputs},
              'files_sha256':{name:file_hash(output/name) for name in ['train.jsonl','verification.jsonl','excluded.jsonl','reserved-mbpp.jsonl']},
              'limitations':['Published examples are small correctness checks, not exhaustive tests or efficiency proofs.',
                             'Only the existing constrained Python subset is executable; unsupported classes/imports are excluded.',
                             'Name variants reuse the same algorithms and cases; they are not additional unique problems.',
                             'Prior corpora have not been fully audited for MBPP benchmark contamination; final independent code tests remain required.']}
    atomic_json(output/'manifest.json',manifest)
    atomic_json(output/'progress.json',{'status':'verified public code supplement prepared; not trained','train_examples':len(rows),'unique_programs':manifest['unique_programs']})
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['output','source-cache','functions','tokenizer','questions']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--variants',type=int,default=4)
    parser.add_argument('--documented-functions');parser.add_argument('--existing-verified');args=parser.parse_args()
    print(json.dumps(prepare(args.output,args.source_cache,args.functions,args.tokenizer,args.questions,args.variants,
                             args.documented_functions,args.existing_verified),indent=2))


if __name__=='__main__':main()
