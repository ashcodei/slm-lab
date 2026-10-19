"""Post-training syntax, hidden-case correctness and empirical efficiency checks."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import re
import subprocess
import sys
import torch
from tokenizers import Tokenizer
from .code_tasks import TASKS,request
from .corpus import file_hash
from .training import load_checkpoint
from .evaluation_provenance import capture, seal

WORKER=Path(__file__).resolve().parents[2]/'scripts/code_worker.py'


def execute(payload):
    # No credentials are inherited by the candidate execution worker.
    env={'PATH':'/usr/bin:/bin:'+str(Path(sys.executable).parent),'LANG':'C.UTF-8',
         'PYTHONHASHSEED':'0'}
    import shutil
    node=shutil.which('node')
    if node:env['PATH']=str(Path(node).parent)+':'+env['PATH']
    try:
        completed=subprocess.run([sys.executable,'-I',str(WORKER)],input=json.dumps(payload),
            text=True,capture_output=True,timeout=14,env=env)
        lines=[line for line in completed.stdout.splitlines() if line.startswith('{')]
        result=json.loads(lines[-1]) if lines else {'syntax_valid':False,'functional_passed':False}
        if completed.returncode:result.update({'efficiency_completed':False,'worker_exit_code':completed.returncode})
        return result
    except subprocess.TimeoutExpired as error:
        output=error.stdout or b''
        if isinstance(output,bytes):output=output.decode(errors='replace')
        lines=[line for line in output.splitlines() if line.startswith('{')]
        result=json.loads(lines[-1]) if lines else {'syntax_valid':False,'functional_passed':False}
        return {**result,'efficiency_completed':False,'error':'Worker deadline exceeded'}


def efficiency(candidate,reference,scaling):
    if not candidate.get('functional_passed') or not candidate.get('efficiency_completed'):return False,{}
    c=candidate['stress'];r=reference['stress'];ratios=[a['median_seconds']/max(b['median_seconds'],1e-7) for a,b in zip(c,r)]
    # Report measured growth; it is evidence, not proof of an asymptotic bound.
    growth=c[-1]['median_seconds']/max(c[0]['median_seconds'],1e-6)
    limit=4 if scaling=='logarithmic' else 40
    runtime=c[-1]['median_seconds']<=max(.025,r[-1]['median_seconds']*30)
    return bool(runtime and growth<=limit and c[-1]['memory_bytes']<=64*1024*1024),{
        'relative_runtime':ratios,'observed_growth':growth,'growth_limit':limit,
        'large_input_runtime_limit_seconds':max(.025,r[-1]['median_seconds']*30)}


def instruction_code(text):
    """Extract a code fence without repairing or rewriting any generated code."""
    fence=re.search(r'```(?:python|javascript|js|py)?[ \t]*\r?\n(.*?)(?:```|\Z)',text,re.S|re.I)
    return fence[1].strip() if fence else text.strip()


@torch.no_grad()
def evaluate(checkpoint,output,device='cuda',max_new_tokens=512,languages=('Python','JavaScript'),prompt_mode='completion'):
    provenance=capture()
    if prompt_mode not in {'completion','instruction'}:raise ValueError('Unknown code prompt mode')
    torch.set_num_threads(2);model,meta=load_checkpoint(checkpoint,device,training_state=False);model.eval()
    tp=Path(checkpoint).parent/'tokenizer.json';tokenizer=Tokenizer.from_file(str(tp))
    if file_hash(tp)!=meta['tokenizer_sha256']:raise ValueError('Tokenizer mismatch')
    amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    results=[]
    for language in languages:
        for task in TASKS:
            name,args,specification,scaling,py,js=task
            prefix=(f'def {name}({args}):\n    """{specification}"""\n' if language=='Python'
                    else f'function {name}({args}) {{\n  // {specification}\n')
            if prompt_mode=='instruction':
                signature=f'def {name}({args}):' if language=='Python' else f'function {name}({args})'
                runtime=('Use built-in Python types; only the math module is available.' if language=='Python'
                         else 'Use JavaScript language built-ins; Node/browser APIs are unavailable.')
                prompt=f'User: Write a {language} function with signature {signature}. {specification}\n{runtime}\nReturn only the complete function and any necessary helper functions.\nAssistant:'
            else:prompt=prefix
            ids=torch.tensor([[tokenizer.token_to_id('<bos>')]+tokenizer.encode(prompt).ids],device=device)
            with amp():generated=model.generate(ids,max_new_tokens=max_new_tokens,eos_token_id=tokenizer.token_to_id('<eos>'),use_cache=True)
            raw_response=tokenizer.decode(generated[0,ids.shape[1]:].tolist())
            completion=raw_response.split('```')[0]
            code=instruction_code(raw_response) if prompt_mode=='instruction' else prefix+completion
            if language=='Python':code=re.split(r'\n(?=File:)',code)[0]
            candidate=execute(request(task,language,code));ref=execute(request(task,language,py if language=='Python' else js,True))
            if not ref.get('functional_passed') or not ref.get('efficiency_completed'):
                raise RuntimeError('Reference evaluator failed: '+str(ref))
            passed,measure=efficiency(candidate,ref,scaling)
            result={'language':language,'task':name,'specification':specification,'code':code,
                'raw_response':raw_response,
                'intended_scaling':scaling,'candidate':candidate,'reference_stress':ref['stress'],
                'efficiency_stress_passed':passed,'efficiency_measurement':measure}
            results.append(result);print(json.dumps({k:result[k] for k in ['language','task','efficiency_stress_passed']}),flush=True)
    report={'checkpoint_sha256':file_hash(checkpoint),'generation_budget':max_new_tokens,'decoding':'greedy, cached BF16' if device.startswith('cuda') else 'greedy, cached FP32',
        'prompt_mode':prompt_mode,
        'task_count':len(results),'syntax_pass_rate':sum(r['candidate'].get('syntax_valid',False) for r in results)/len(results),
        'functional_pass_at_1':sum(r['candidate'].get('functional_passed',False) for r in results)/len(results),
        'efficiency_stress_pass_rate':sum(r['efficiency_stress_passed'] for r in results)/len(results),
        'by_language':{lang:{'tasks':sum(r['language']==lang for r in results),
            'syntax_passed':sum(r['language']==lang and r['candidate'].get('syntax_valid',False) for r in results),
            'functional_passed':sum(r['language']==lang and r['candidate'].get('functional_passed',False) for r in results),
            'efficiency_passed':sum(r['language']==lang and r['efficiency_stress_passed'] for r in results)} for lang in languages},
        'results':results,'suite_sha256':file_hash(Path(__file__).with_name('code_tasks.py')),
        'limitations':['Twelve local tasks per language; not a standard leaderboard score or coverage of all code.',
            'Python execution uses a constrained subset with public math imports; JavaScript has no host bindings.',
            'Timing and memory stress tests do not prove optimal asymptotic complexity.',
            'Task concepts may overlap with algorithms in training; prompts/tests/reference programs were held out.']}
    seal(report,provenance,device=device,tokenizer_sha256=meta['tokenizer_sha256'])
    Path(output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--max-new-tokens',type=int,default=512)
    p.add_argument('--prompt-mode',choices=['completion','instruction','both'],default='both');a=p.parse_args()
    modes=['completion','instruction'] if a.prompt_mode=='both' else [a.prompt_mode]
    for mode in modes:
        output=Path(a.output)
        if a.prompt_mode=='both' and mode=='instruction':output=output.with_stem(output.stem+'-instruction')
        evaluate(a.checkpoint,str(output),a.device,a.max_new_tokens,prompt_mode=mode)

if __name__=='__main__':main()
