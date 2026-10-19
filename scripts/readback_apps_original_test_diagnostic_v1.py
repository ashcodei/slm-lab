"""Recount original held diagnostic delivery, complete raw cases and repetition."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re

from tokenizers import Tokenizer

from slm_lab.apps_published_cases_v2 import contract
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'reports/APPS-original-test-diagnostic-v1'
DATA=ROOT/'data/public-apps-publisher-test-evaluation-v1'
TOKEN=ROOT/'checkpoints/dialogue-retention-v1/systems-variety-code/tokenizer.json'


def repeated_span(tokens):
    longest=0
    for period in range(1,33):
        run=0
        for i in range(period,len(tokens)):
            run=run+1 if tokens[i]==tokens[i-period] else 0
            if run>=4*period:
                longest=max(longest,run+period)
    return longest


def main():
    output=FOLDER/'independent-results-and-repetition-readback.json'
    if output.exists():
        raise ValueError('Preserve original diagnostic readback')
    plan=json.loads((FOLDER/'plan.json').read_text());summary=json.loads((FOLDER/'comparison.json').read_text())
    tokenizer=Tokenizer.from_file(str(TOKEN))
    sources={r['id']:r for r in map(json.loads,(DATA/'evaluation-records.jsonl').read_text().splitlines())}
    if file_hash(TOKEN)!=plan['tokenizer_sha256'] or summary['plan_sha256']!=file_hash(FOLDER/'plan.json'):
        raise ValueError('Original actual diagnostic protocol required')
    result={};proofs=[];paths=[Path(__file__).resolve(),FOLDER/'plan.json',FOLDER/'comparison.json',TOKEN]
    for phase in ('baseline','candidate'):
        records=[];counts=Counter();outcomes=Counter();periodic=[]
        for shard in range(3):
            path=FOLDER/f'{phase}-{shard}.jsonl';certificate=FOLDER/f'{phase}-{shard}.json'
            receipt=json.loads(certificate.read_text())
            if (file_hash(path)!=receipt['journal_sha256'] or receipt['plan_sha256']!=file_hash(FOLDER/'plan.json')
                    or receipt['checkpoint_sha256']!=summary[phase]['checkpoint_sha256']):
                raise ValueError('Exact fixed policy, complete worker and journal bytes required')
            rows=list(map(json.loads,path.read_text().splitlines()))
            if len(rows)!=receipt['tasks']:
                raise ValueError('Every actual original worker task required')
            records.extend(rows);paths.extend([path,certificate])
        if sorted(r['id'] for r in records)!=plan['selected_original_test_ids']:
            raise ValueError('Every source-selected original problem exactly once required')
        for row in records:
            original=sources[row['id']];tokens=row['generation_token_ids']
            prefix=[1,*tokenizer.encode('User: '+original['complete_prompt']+'\nAssistant:').ids]
            prefix_sha=hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest()
            if (not 0<len(tokens)<=1024 or len(prefix)+1024>4096 or prefix_sha!=row['prompt_input_ids_sha256']
                    or prefix_sha!=original['complete_prompt_input_ids_sha256'] or tokenizer.decode(tokens)!=row['raw_response']
                    or (len(tokens)<1024 and tokens[-1]!=2) or row['problem']!=original['problem'] or row['mode']!=original['mode']):
                raise ValueError('All actual original prompts and generated token/response bytes must reconstruct')
            response=row['raw_response'];fence=re.fullmatch(r'\s*```(?:python|py)?[ \t]*\r?\n(.*?)\r?\n```\s*',response,re.S|re.I)
            code=(fence[1] if fence else None) if response.lstrip().startswith('```') else (response.strip() or None)
            if code!=row['extracted_code']:
                raise ValueError('Actual complete delivered program must reconstruct')
            observed=row['execution'];syntax=False
            if code:
                try:
                    compile(code,'<original-held-diagnostic>','exec');syntax=True
                except (SyntaxError,ValueError,RecursionError):
                    pass
                if hashlib.sha256(code.encode()).hexdigest()!=observed['original_program_sha256']:
                    raise ValueError('Actual complete delivered program hash changed')
            if syntax!=bool(observed['syntax_valid']):
                raise ValueError('Actual delivered Python compilation must reproduce syntax score')
            path=DATA/'original-test'/row['problem']/'input_output.json'
            cases=contract(path.read_bytes())
            if (file_hash(path)!=row['original_tests_sha256'] or cases['total_cases']!=observed['total_cases']
                    or cases['mode']!=row['mode']):
                raise ValueError('Every original complete source case set must match')
            if observed['all_cases_observed']:
                raw=observed['raw_results']['results']
                if len(raw)!=cases['total_cases'] or observed['execution']['status']!='completed' or observed['execution']['returncode']!=0:
                    raise ValueError('All original case results and completed execution required')
                flags=[]
                for actual,expected in zip(raw,cases['outputs']):
                    value=actual['value']
                    flag=(isinstance(value,str) and value.strip()==expected.strip()) if cases['mode']=='stdin' else (
                        value==expected or (isinstance(expected,list) and bool(expected) and value==expected[0]))
                    flags.append(flag)
                if flags!=observed['case_passes'] or sum(flags)!=observed['passed_cases'] or all(flags)!=observed['all_finite_published_cases_pass']:
                    raise ValueError('Every actual original expected/observed case comparison must reconstruct')
                counts['completed_case_programs']+=1;counts['completed_individual_cases']+=len(flags)
                counts['passed_individual_cases']+=sum(flags)
            elif observed['all_finite_published_cases_pass']:
                raise ValueError('No complete program pass without all original case observations')
            span=repeated_span(tokens);periodic.append(span)
            counts['syntax_valid']+=syntax;counts['whole_program_case_passes']+=observed['all_finite_published_cases_pass']
            counts['at_1024_token_budget']+=len(tokens)==1024;counts['periodic_span_at_least64_tokens']+=span>=64
            outcomes[observed['status']]+=1
            proofs.append(dict(phase=phase,id=row['id'],generation_tokens=len(tokens),largest_periodic_span=span,
                complete_cases_observed=observed['all_cases_observed']))
        if (counts['syntax_valid']!=summary[phase]['syntax_valid']
                or counts['whole_program_case_passes']!=summary[phase]['all_finite_published_cases_pass']
                or dict(outcomes)!=summary[phase]['outcomes']):
            raise ValueError('Actual raw outcomes must reproduce original aggregate')
        result[phase]=dict(tasks=len(records),**dict(counts),outcomes=dict(outcomes),
            largest_periodic_span=max(periodic),mean_largest_periodic_span=sum(periodic)/len(periodic))
    atomic_json(output,dict(completed_at=datetime.now(timezone.utc).isoformat(),**result,
        every_original_selected_prompt_generated_token_response_delivered_program_and_case_result_reconstructed=True,
        repetition_scope='Longest exactly periodic token span, period1..32, at least5repeats; diagnostic, not proof of semantic error.',
        runtime_scope='Original retained bounded local stdlib/class/stdin execution; no new candidate execution and not official APPS runtime.',
        runtime_or_resource_qualifications_not_labeled_wrong_answers=True,
        diagnostic_not_full_or_certified_independent_APPS=True,per_original_program=proofs,
        evidence_files_sha256={str(p):file_hash(p) for p in paths},model_updates=0,goal_achieved=False))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
