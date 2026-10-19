"""Reconcile actual edits/cases/targets and screen named held-source matches."""
import argparse
import ast
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import time
import tokenize

from tokenizers import Tokenizer

from audit_complete_apps_candidate_v2 import digest
from slm_lab.apps_execution_repair_v1 import OPS
from slm_lab.apps_published_cases_v2 import PRELUDE,contract
from slm_lab.code_tasks import TASKS
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/APPS-execution-repairs-v1'
REPORTS=ROOT/'reports/APPS-execution-repairs-v1'
PARENT=ROOT/'reports/APPS-original-complete-cases-v2'
SOURCE=ROOT/'data/public-apps-original-inventory-v1'


def same_process(pid,ticks):
    path=Path('/proc')/str(pid)/'stat'
    if not path.exists():return False
    fields=path.read_text().split(') ',1)[1].split()
    return fields[19]==ticks and fields[0] not in {'Z','X'}


def flags(observed,cases):
    values=observed['raw_results']['results']
    if (observed['total_cases']!=cases['total_cases'] or len(values)!=cases['total_cases']
            or observed['original_tests_sha256']!=cases['original_tests_sha256']
            or observed['mode']!=cases['mode'] or not observed['all_cases_observed']
            or observed['execution']['status']!='completed' or observed['execution']['returncode']!=0):
        raise ValueError('Every original source case/result and successful bounded execution required')
    computed=[]
    for row,expected in zip(values,cases['outputs']):
        actual=row['value']
        if cases['mode']=='stdin':
            computed.append(isinstance(actual,str) and actual.strip()==expected.strip())
        else:
            computed.append(actual==expected or (isinstance(expected,list) and bool(expected) and actual==expected[0]))
    if (computed!=observed['case_passes'] or sum(computed)!=observed['passed_cases']
            or all(computed)!=observed['all_finite_published_cases_pass']):
        raise ValueError('Actual complete original expected values must reproduce every comparison')
    return computed


def source_observations(plan):
    positives={r['id']:r for r in map(json.loads,(PARENT/'observations.jsonl').read_text().splitlines())
        if r['status']=='all_finite_published_cases_pass'}
    selections={r['id']:r for r in map(json.loads,(DATA/'selected-mutations.jsonl').read_text().splitlines())}
    if selections.keys()!=positives.keys() or len(positives)!=plan['source_passing_original_programs']:
        raise ValueError('Every original passing source selection required')
    view=ROOT/'data/public-apps-human-solution-view-v1'
    train={r['id']:r for r in map(json.loads,(view/'train-records.jsonl').read_text().splitlines())}
    held_groups={r['group_id'] for r in map(json.loads,(view/'validation-records.jsonl').read_text().splitlines())}
    originals={};source_cases={};planned={};token_spans={}
    for identity,positive in positives.items():
        if (identity not in train or train[identity]['split']!='train' or train[identity]['group_id'] in held_groups
                or train[identity]['source_problem_id']!=positive['problem']):
            raise ValueError('Original admitted train source identity/group required')
        folder=SOURCE/'original-train'/positive['problem']
        program=json.loads((folder/'solutions.json').read_bytes())[positive['original_solution_index']]
        if hashlib.sha256(program.encode()).hexdigest()!=positive['original_program_sha256']:
            raise ValueError('Actual complete original human bytes differ')
        originals[identity]=program
        cases=contract((folder/'input_output.json').read_bytes());source_cases[identity]=cases
        if not all(flags(positive,cases)):
            raise ValueError('Every actual original program must pass every complete original source case')
        starts=[0]
        for line in program.splitlines(keepends=True):starts.append(starts[-1]+len(line))
        token_spans[identity]={(starts[t.start[0]-1]+t.start[1],starts[t.end[0]-1]+t.end[1],t.string)
            for t in tokenize.generate_tokens(io.StringIO(program).readline)
            if t.type in {tokenize.OP,tokenize.NAME,tokenize.NUMBER} and t.string in OPS}
        for mutation in selections[identity]['mutations']:
            key=(identity,mutation['ordinal'])
            if key in planned:raise ValueError('Every preselected edit identity must be unique')
            planned[key]=mutation
    if len(planned)!=plan['source_selected_mutations']:
        raise ValueError('Every original predeclared mutation required')
    observations=list(map(json.loads,(DATA/'observations.jsonl').read_text().splitlines()))
    seen=set();witnesses={};outcomes=Counter()
    for record in observations:
        identity=record['original_id'];mutation=record['mutation'];key=(identity,mutation['ordinal'])
        if key not in planned or key in seen:raise ValueError('Unexpected/repeated original mutation result')
        seen.add(key);patch=mutation['patch'];program=originals[identity];cases=source_cases[identity]
        if (record['publisher_split']!='train' or record['human_preferences'] or record['model_generated']
                or record['external_model_teacher_used']):raise ValueError('Actual train-only machine execution source required')
        if ({k:mutation[k] for k in ['ordinal','patch','program_sha256']}!=planned[key]
                or (patch['start'],patch['end'],patch['old']) not in token_spans[identity]
                or OPS[patch['old']]!=(patch['new'],patch['family'])):
            raise ValueError('Actual edit must be exactly one declared original Python token')
        expected=program[:patch['start']]+patch['new']+program[patch['end']:]
        if expected!=mutation['program'] or hashlib.sha256(expected.encode()).hexdigest()!=mutation['program_sha256']:
            raise ValueError('Every actual executed mutation byte must reconstruct')
        observed=record['execution'];outcomes[observed['status']]+=1
        if observed['original_program_sha256']!=mutation['program_sha256']:
            raise ValueError('Actual worker must execute the exact predeclared mutation')
        computed=flags(observed,cases) if observed.get('all_cases_observed') else None
        valid=observed['status']=='completed_published_case_mismatch' and computed is not None and not all(computed)
        if record['source_counterexample_observed']!=valid or (record['witness'] is not None)!=valid:
            raise ValueError('Only actual completed wrong values can supply repair labels')
        if not valid:continue
        compile(mutation['program'],'<audited-mutation>','exec')
        i=computed.index(False)
        witness=dict(case_index=i,complete_original_case_count=cases['total_cases'],mode=cases['mode'],function=cases['function'],
            input=cases['inputs'][i],original_expected=cases['outputs'][i],observed=observed['raw_results']['results'][i]['value'],
            original_tests_sha256=cases['original_tests_sha256'])
        if witness!=record['witness']:raise ValueError('First concrete source counterexample must reconstruct exactly')
        witnesses[identity+':execution-repair-v1:'+str(mutation['ordinal'])]=record
    if seen!=planned.keys():raise ValueError('All predeclared original edits must have an actual retained outcome')
    return train,originals,source_cases,witnesses,outcomes,len(observations)


def held_index():
    result=defaultdict(set)
    mbpp=ROOT/'data/verified-code-source-cache-v1/sanitized-mbpp.json'
    codenet=ROOT/'data/codenet-durable-pack-v2/original-rows.jsonl.gz'
    apps=ROOT/'data/public-apps-complete-pack-v2/original-rows.jsonl.gz'
    for row in json.loads(mbpp.read_text()):
        if 11<=row['task_id']<=510:
            for code in [row['code'],'\n'.join(row.get('test_imports',[]))+'\n'+row['code']]:
                result[digest(code)].add('sanitized-MBPP-publisher-test')
    for task in TASKS:result[digest(task[4])].add('local-Python12-development-suite')
    for path,label in [(codenet,'CodeNet'),(apps,'APPS-internal-validation')]:
        with gzip.open(path,'rt') as stream:
            for row in map(json.loads,stream):
                if row['split']=='train' or 'python' not in row['language'].lower():continue
                try:normalized=digest(row['response'])
                except (SyntaxError,ValueError,RecursionError):continue
                result[normalized].add(label+':'+row['split'])
    return result,[mbpp,codenet,apps,ROOT/'src/slm_lab/code_tasks.py',ROOT/'scripts/audit_complete_apps_candidate_v2.py']


def check_rows(train,originals,witnesses,held):
    tokenizer=Tokenizer.from_file(str(DATA/'tokenizer.json'))
    pack=CompleteCodePack(DATA,'train',purpose='audit')
    if len(pack)!=pack.manifest['source_examples'] or any(pack.manifest['counts'][r]['examples'] for r in ['validation','test']):
        raise ValueError('Only explicit original train source rows may enter this candidate')
    reviews=list(map(json.loads,(DATA/'reviewed-repairs.jsonl').read_text().splitlines()))
    with gzip.open(DATA/'original-rows.jsonl.gz','rt') as stream:rows=list(map(json.loads,stream))
    if len(rows)!=len(pack):raise ValueError('Every actual candidate pack row required')
    seen=set();matched=[];families=Counter();problems=set();targets=0
    for row,in_pack in [(r,True) for r in rows]+[(r,False) for r in reviews]:
        identity=row['id']
        if identity not in witnesses or identity in seen:raise ValueError('Every actual finite repair row must be unique')
        seen.add(identity);record=witnesses[identity];source_id=record['original_id'];program=originals[source_id]
        original=train[source_id];base=SOURCE/'original-train'/original['source_problem_id']
        question=(base/'question.txt').read_bytes().decode()
        starter=(base/'starter_code.py').read_bytes().decode() if (base/'starter_code.py').exists() else None
        if (row['response']!=program or row['split']!='train' or row['publisher_split']!='train'
                or row['group_id']!=original['group_id'] or row['human_preferences'] or row['external_model_teacher_used']
                or row['original_target_program_bytes_modified'] or row['mutation']!=record['mutation']
                or row['witness']!=record['witness'] or row['original_source_program_id']!=source_id
                or row['original_source_body_sha256']!=hashlib.sha256(program.encode()).hexdigest()
                or row['original_question_sha256']!=hashlib.sha256(question.encode()).hexdigest()):
            raise ValueError('Every actual repair target/role/edit/witness must match complete original sources')
        display=json.dumps(dict(input=row['witness']['input'],expected=row['witness']['original_expected'],
            actual=row['witness']['observed']),ensure_ascii=False,sort_keys=True,separators=(',',':'))
        if (question not in row['prompt'] or (starter and starter not in row['prompt'])
                or row['mutation']['program'] not in row['prompt'] or display not in row['prompt'] or PRELUDE not in row['prompt']):
            raise ValueError('Complete original problem/declared runtime/bad program/exact counterexample required')
        if row['witness']['mode']=='function' and row['witness']['function'] not in row['prompt']:
            raise ValueError('Actual original function interface must be present in the repair prompt')
        prefix=[tokenizer.token_to_id('<bos>')]+tokenizer.encode('User: '+row['prompt']+'\nAssistant:').ids
        body=tokenizer.encode(program).ids+[tokenizer.token_to_id('<eos>')]
        actual_ids=prefix+body;actual_labels=[-100]*len(prefix)+body
        if row['complete_context_tokens']!=len(actual_ids) or row['complete_response_targets']!=len(body):
            raise ValueError('Complete original prompt/human body/EOS token counts must reconcile')
        if in_pack:
            position=row['token_pack_index']
            if pack.sequence(position)!=(actual_ids,actual_labels) or len(actual_ids)>4096 or row['review_reasons']:
                raise ValueError('Every actual complete candidate token and response/EOS mask must reconstruct')
            families[row['mutation']['patch']['family']]+=1;problems.add(row['source_problem_id']);targets+=len(body)
            reasons=[];tags=set()
            for role,code in [('target',program),('prompt_mutant',row['mutation']['program'])]:
                try:normalized=digest(code)
                except (SyntaxError,ValueError,RecursionError):reasons.append(role+':normalization_requires_review');continue
                tags.update(role+':'+name for name in held.get(normalized,[]))
            if tags or reasons:matched.append(dict(id=identity,held_sources=sorted(tags),reviews=reasons))
        elif not row['review_reasons'] or (len(actual_ids)<=4096 and not any(x in row['prompt'] or x in program for x in ['<pad>','<bos>','<eos>'])):
            raise ValueError('Every whole-context review needs an actual preserved length/marker reason')
    if seen!=witnesses.keys():raise ValueError('Every actual complete counterexample must be a whole candidate or a preserved review')
    return dict(complete_fit_repair_examples=len(rows),preserved_whole_context_reviews=len(reviews),
        complete_response_targets=targets,unique_source_problems=len(problems),mutation_families=dict(families),
        actual_unique_original_human_targets=len({r['original_source_program_id'] for r in rows}),
        all_actual_repair_rows_tokens_and_response_EOS_masks_reconstructed=True,
        all_targets_byte_equal_original_human_programs=True,exact_named_held_matches_or_normalization_reviews=matched)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-pid',type=int)
    parser.add_argument('--wait-for-ticks')
    args=parser.parse_args()
    if bool(args.wait_for_pid)!=bool(args.wait_for_ticks):raise ValueError('Exact collection process identity required')
    completed=REPORTS/'candidate-complete.json';target=REPORTS/'independent-candidate-readback.json'
    if target.exists():raise ValueError('Preserve earlier independent repair readback')
    while not completed.exists():
        if not args.wait_for_pid or not same_process(args.wait_for_pid,args.wait_for_ticks):
            raise ValueError('Original collection is terminal without complete evidence; preserve partial artifacts')
        time.sleep(5)
    plan=json.loads((REPORTS/'plan.json').read_text());summary=json.loads(completed.read_text())
    for name,expected in plan['source_files_sha256'].items():
        if file_hash(name)!=expected:raise ValueError('Original bound source/runtime changed')
    for path,key in [(DATA/'selected-mutations.jsonl','selected_mutations_sha256'),
            (DATA/'observations.jsonl','observations_sha256'),(DATA/'reviewed-repairs.jsonl','reviewed_repairs_sha256'),
            (DATA/'manifest.json','candidate_manifest_sha256'),(REPORTS/'plan.json','source_plan_sha256')]:
        if file_hash(path)!=summary[key]:raise ValueError('Actual complete candidate bytes differ')
    train,originals,cases,witnesses,outcomes,observations=source_observations(plan)
    held,anchors=held_index()
    proof=check_rows(train,originals,witnesses,held)
    counts=summary['counts']
    if (observations!=counts['executed_mutations'] or len(witnesses)!=counts['complete_finite_counterexamples']
            or proof['complete_fit_repair_examples']!=counts['complete_fit_repair_examples']
            or proof['preserved_whole_context_reviews']!=counts.get('reviewed_whole_repair_context',0)
            or proof['complete_response_targets']!=summary['complete_original_response_targets']
            or dict(outcomes)!=summary['execution_outcomes']):
        raise ValueError('Every actual source/counterexample/whole-target population must reconcile')
    paths=[Path(__file__).resolve(),*anchors,DATA/'manifest.json',DATA/'selected-mutations.jsonl',
        DATA/'observations.jsonl',DATA/'reviewed-repairs.jsonl',REPORTS/'plan.json',completed]
    result=dict(completed_at=datetime.now(timezone.utc).isoformat(),source_original_passing_programs=len(originals),
        actual_executed_source_mutations=observations,actual_complete_finite_counterexamples=len(witnesses),
        execution_outcomes=dict(outcomes),**proof,
        every_original_train_identity_and_all_case_result_independently_reconciled=True,
        every_actual_single_token_edit_and_first_failed_value_independently_reconstructed=True,
        source_groups_crossing_internal_validation=0,original_target_or_case_bytes_clipped=0,
        held_screen_scope=['All retained sanitized-MBPP test references in IDs11..510, with/without original imports.',
            'All12local Python development references.', 'All Python3-parseable original CodeNet validation/test programs.',
            'All12068original APPS internal-validation human programs.'],
        exact_screen_not_semantic_or_prior_public_corpus_independence_certificate=True,
        candidate_manifest_sha256=file_hash(DATA/'manifest.json'),
        files_sha256={str(p):file_hash(p) for p in paths},
        candidate_not_admitted=True,training_admitted=False,human_preferences=False,
        external_model_teacher_used=False,model_generated_examples=0,model_updates=0,
        algorithmic_correctness_or_efficiency_proven=False,model_capability_gain=False,goal_achieved=False)
    atomic_json(target,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['files_sha256','exact_named_held_matches_or_normalization_reviews']},indent=2),flush=True)


if __name__=='__main__':main()
