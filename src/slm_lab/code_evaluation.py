"""Small Python function-completion check with restricted, bounded execution.
This is a local smoke benchmark, not evidence of general coding capability.
"""
import argparse
import ast
from contextlib import nullcontext
import json
from pathlib import Path
import re
import subprocess
import sys
import torch
from tokenizers import Tokenizer
from .corpus import file_hash
from .training import load_checkpoint

TASKS=[
 ('sum_even','xs','Return the sum of the even integers in xs.',[([1,2,3,4],6),([],0),([-2,1,8],6)]),
 ('count_positive','xs','Return how many elements of xs are strictly positive.',[([-1,0,2,4],2),([],0),([0,0],0)]),
 ('clamp_zero','n','Return n when n is positive, otherwise return zero.',[(5,5),(-4,0),(0,0)]),
 ('square_sum','xs','Return the sum of the squares of every integer in xs.',[([2,3],13),([],0),([-2,1],5)]),
 ('factorial','n','Return n factorial for a nonnegative integer n.',[(0,1),(1,1),(5,120)]),
 ('is_palindrome','s','Return whether string s is a palindrome.',[('abba',True),('abc',False),('',True)]),
 ('last_or_zero','xs','Return the last element of list xs, or zero when xs is empty.',[([2,9],9),([],0),([-1],-1)]),
 ('sorted_unique','xs','Return the sorted unique integers of xs as a list.',[([3,1,3,2],[1,2,3]),([],[]),([-1,-1],[-1])])]

# The worker is our trusted code; generated text is supplied as JSON data.
WORKER=r'''
import ast,json,resource,sys
resource.setrlimit(resource.RLIMIT_CPU,(2,2))
resource.setrlimit(resource.RLIMIT_AS,(256*1024*1024,256*1024*1024))
resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
request=json.load(sys.stdin)
try:
 tree=ast.parse(request['code'])
 banned=(ast.Import,ast.ImportFrom,ast.Attribute,ast.ClassDef,ast.Global,ast.Nonlocal,ast.With,ast.AsyncWith)
 if any(isinstance(n,banned) or (isinstance(n,ast.Name) and n.id.startswith('__')) for n in ast.walk(tree)):
  raise ValueError('Outside restricted Python subset')
 if not tree.body or any(not isinstance(n,ast.FunctionDef) or n.decorator_list for n in tree.body):
  raise ValueError('Only undecorated function definitions allowed at module level')
 allowed={k:__builtins__.__dict__[k] for k in ['range','len','sum','sorted','min','max','abs','enumerate','zip','reversed','list','tuple','set','dict','int','float','str','bool','all','any','pow','divmod']}
 namespace={'__builtins__':allowed}
 exec(compile(tree,'<generated>','exec'),namespace)
 fn=namespace[request['function']]
 passed=sum(fn(value)==expected for value,expected in request['cases'])
 print(json.dumps({'passed_cases':passed,'total_cases':len(request['cases']),'passed':passed==len(request['cases'])}))
except BaseException as error:print(json.dumps({'passed':False,'error':type(error).__name__+': '+str(error)[:200]}))
'''


def check(code,function,cases):
    try:
        process=subprocess.run([sys.executable,'-I','-c',WORKER],
            input=json.dumps({'code':code,'function':function,'cases':cases}),text=True,
            capture_output=True,timeout=5)
        return json.loads(process.stdout) if process.returncode==0 else {'passed':False,'error':'Worker resource limit or failure'}
    except Exception as error:return {'passed':False,'error':str(error)[:200]}


@torch.no_grad()
def evaluate(checkpoint,output,device='cuda',max_new_tokens=128):
    torch.set_num_threads(4);model,meta=load_checkpoint(checkpoint,device,training_state=False);model.eval()
    tokenizer_path=Path(checkpoint).parent/'tokenizer.json';tokenizer=Tokenizer.from_file(str(tokenizer_path))
    if meta['tokenizer_sha256']!=file_hash(tokenizer_path):raise ValueError('Tokenizer mismatch')
    amp=torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    records=[]
    with amp:
        for name,arg,description,cases in TASKS:
            prompt=f'def {name}({arg}):\n    """{description}"""\n'
            ids=torch.tensor([[tokenizer.token_to_id('<bos>')]+tokenizer.encode(prompt).ids],device=device)
            generated=model.generate(ids,max_new_tokens=max_new_tokens,eos_token_id=tokenizer.token_to_id('<eos>'))
            completion=tokenizer.decode(generated[0,ids.shape[1]:].tolist())
            # Stop before a subsequent file/module-level declaration.
            completion=re.split(r'\n(?=def |class |if __name__|File:)',completion)[0]
            code=prompt+completion
            try:ast.parse(code);syntax=True
            except SyntaxError:syntax=False
            result=check(code,name,cases) if syntax else {'passed':False,'error':'SyntaxError'}
            records.append({'task':name,'syntax_valid':syntax,'completion':completion,**result})
    report={'checkpoint_sha256':file_hash(checkpoint),'tasks':len(records),'max_new_tokens':max_new_tokens,
        'syntax_valid':sum(r['syntax_valid'] for r in records),'passed':sum(r['passed'] for r in records),
        'greedy_pass_at_1':sum(r['passed'] for r in records)/len(records),'results':records,
        'limitations':['Eight elementary Python completions only; not a multilingual benchmark.',
            'Imports, attribute access and other Python constructs are deliberately rejected by execution checks.',
            'Tests were not supplied to the model; conceptual overlap with algorithms in training is possible.']}
    Path(output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--max-new-tokens',type=int,default=128);a=p.parse_args()
    evaluate(a.checkpoint,a.output,a.device,a.max_new_tokens)

if __name__=='__main__':main()
