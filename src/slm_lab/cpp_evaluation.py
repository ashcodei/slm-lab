"""C++ generation assessment with kernel-constrained compile/runtime workers."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import re

import torch
from tokenizers import Tokenizer

from .code_tasks import TASKS,request
from .corpus import file_hash
from .cpp_tasks import SIGNATURES,reference
from .native_code import execute,efficiency,toolchain
from .training import load_checkpoint


def extract(text):
    match=re.search(r'```(?:cpp|c\+\+|cc|cxx)?[ \t]*\r?\n(.*?)(?:```|\Z)',text,re.S|re.I)
    return match[1].strip() if match else text.strip()


def verify_references(output):
    tools=toolchain();records=[]
    for task in TASKS:
        result=execute(request(task,'C++',reference(task[0]),True))
        if not result.get('functional_passed') or not result.get('efficiency_completed'):
            raise RuntimeError('Native reference failed: '+task[0]+' '+str(result))
        records.append({'task':task[0],'result':result})
        print('native reference verified',task[0],flush=True)
    report={'toolchain':tools,'tasks':len(records),'references':records,
            'scope':'Reference/runner validation only; this is not a trained-model capability score.'}
    Path(output).write_text(json.dumps(report,indent=2)+'\n');return report


@torch.no_grad()
def evaluate(checkpoint,output,device='cuda',max_new_tokens=512):
    from .evaluation_provenance import capture,seal
    provenance=capture()
    torch.set_num_threads(1);tools=toolchain()
    model,meta=load_checkpoint(checkpoint,device,training_state=False);model.eval()
    tokenizer_path=Path(checkpoint).parent/'tokenizer.json'
    if file_hash(tokenizer_path)!=meta['tokenizer_sha256']:raise ValueError('Tokenizer mismatch')
    tokenizer=Tokenizer.from_file(str(tokenizer_path));results=[]
    amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    for task in TASKS:
        name,_,description,scaling,_,_=task
        signature=SIGNATURES[name]
        prompt=f'User: Write an ISO C++17 function with signature {signature}. {description}\nInt means long long; Ints means std::vector<Int>. Strings use std::u32string/char32_t, one element per Unicode BMP character. Use std::nullopt for missing optional results. Standard library headers and these aliases are supplied. No main function, nonstandard headers, OS APIs or threads. Return only the complete function and helpers.\nAssistant:'
        ids=torch.tensor([[tokenizer.token_to_id('<bos>')]+tokenizer.encode(prompt).ids],device=device)
        with amp():tokens=model.generate(ids,max_new_tokens=max_new_tokens,eos_token_id=tokenizer.token_to_id('<eos>'))
        raw=tokenizer.decode(tokens[0,ids.shape[1]:].tolist());code=extract(raw)
        candidate=execute(request(task,'C++',code))
        baseline=execute(request(task,'C++',reference(name),True))
        if not baseline.get('functional_passed') or not baseline.get('efficiency_completed'):
            raise RuntimeError('C++ reference runner failed: '+name)
        passed,measurement=efficiency(candidate,baseline)
        results.append({'task':name,'specification':description,'signature':signature,'raw_response':raw,'code':code,
                        'candidate':candidate,'reference_stress':baseline['stress'],
                        'intended_scaling':scaling,'efficiency_stress_passed':passed,'efficiency_measurement':measurement})
        print(json.dumps({'task':name,'syntax':candidate['syntax_valid'],'functional':candidate['functional_passed']}),flush=True)
    report={'checkpoint_sha256':file_hash(checkpoint),'toolchain':tools,'task_count':len(results),
            'syntax_pass_rate':sum(row['candidate']['syntax_valid'] for row in results)/len(results),
            'functional_pass_at_1':sum(row['candidate']['functional_passed'] for row in results)/len(results),
            'efficiency_stress_pass_rate':sum(row['efficiency_stress_passed'] for row in results)/len(results),
            'generation_budget':max_new_tokens,'prompt_mode':'native instruction','results':results,
            'limitations':['Twelve local algorithm tasks, not coverage of all C++ or repository/API work.',
                           'Single-process ISO C++17 function contracts; kernel restrictions intentionally exclude OS/network/thread tasks.',
                           'Hidden edge/random cases and empirical stress do not prove universal correctness or asymptotic optimality.',
                           'Native timing includes transport and input conversion; reference-relative measurements are kept separate from Python/JS scores.']}
    seal(report,provenance,device=device,tokenizer_sha256=meta['tokenizer_sha256'])
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='results'},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-references',action='store_true')
    parser.add_argument('--checkpoint');parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--max-new-tokens',type=int,default=512)
    args=parser.parse_args()
    if args.verify_references:verify_references(args.output)
    else:
        if not args.checkpoint:parser.error('A trained checkpoint is required')
        evaluate(args.checkpoint,args.output,args.device,args.max_new_tokens)


if __name__=='__main__':main()
