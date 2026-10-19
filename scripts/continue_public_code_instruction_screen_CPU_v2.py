"""Continue only the eight unevaluated source rows after a disk-reserve stop.

The failed producer and every original observation remain unchanged. All
1,016 prior complete rows rebuild against pinned inputs before reuse; the
subsequent fresh reader re-executes every surviving whole candidate.
"""
import ast
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from tokenizers import Tokenizer
from qualify_public_code_instruction_candidate_CPU_v1 import sha,read,atomic
from slm_lab.public_code_instruction_screen_v1 import parse_whole,observe,UnsupportedExample
from slm_lab.public_reasoning_protected_text_v1 import population,overlaps

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/public-code-instruction-candidate-v1'
OLD=ROOT/'reports/public-code-instruction-qualification-CPU-v1'
OUT=ROOT/'reports/public-code-instruction-qualification-CPU-v2'
TOKENIZER=ROOT/'data/public-reasoning-curriculum-v1/tokenizer.json'
RESERVE=40*1024**3


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists()
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            or shutil.disk_usage(ROOT).free<RESERVE+128*1024**2
            or read(OLD/'failure.json')['error']!="ValueError('Original writer reserve required')"):
        raise ValueError('Fresh bounded CPU continuation after recovered original reserve required')
    with (OLD/'all1024-original-dispositions.jsonl').open() as stream:
        lines=stream.readlines()
    if len(lines)!=1016 or any(not line.endswith('\n') for line in lines):raise ValueError('Only the exact observed complete1016row prefix may be reused')
    previous=list(map(json.loads,lines));oldplan=read(OLD/'plan.json')
    files=dict(oldplan['source_files_sha256'])
    files.update({str(p):sha(p) for p in [Path(__file__),OLD/'plan.json',OLD/'failure.json',
        OLD/'all1024-original-dispositions.jsonl',OLD/'protected-overlap-configuration.json']})
    if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('All original failed producer/source bytes must remain unchanged')
    metadata={p:(Path(p).stat().st_dev,Path(p).stat().st_ino,Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in files}
    def checked():
        if any((Path(p).stat().st_dev,Path(p).stat().st_ino,Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=m for p,m in metadata.items()):
            raise ValueError('A pinned original byte changed during continuation')
        if shutil.disk_usage(ROOT).free<RESERVE:raise ValueError('Original writer reserve required')
    OUT.mkdir()
    atomic(OUT/'plan.json',dict(source_files_sha256=files,prior_failed_prefix_retained_unchanged=True,
        previous_complete_dispositions=1016,actual_missing_source_rows=8,all_original_source_rows=1024,
        original_reserve_bytes=RESERVE,no_neural_worker_or_producer_restarted=True,
        training_admitted=False,new_neural_updates=0,goal_achieved=False))
    try:
        with (DATA/'first1024-original-entries.jsonl').open() as stream:entries=list(map(json.loads,stream))
        if len(entries)!=1024:raise ValueError('All1024original sources required')
        scanner=[dict(id=str(r['original_shard_position']),prompt=r['original_entry']['input'],
            independent_answer=r['original_entry']['output'],original_entry=dict(question=r['original_entry']['input'])) for r in entries]
        hits,settings=overlaps(scanner,population(ROOT),checked=checked)
        if settings!=read(OLD/'protected-overlap-configuration.json'):raise ValueError('All31349protected inputs must reconstruct')
        atomic(OUT/'protected-overlap-configuration.json',settings)
        tokenizer=Tokenizer.from_file(str(TOKENIZER));counts=Counter();targets=0;contexts=0;new_runs=0
        with (OUT/'all1024-original-dispositions.jsonl').open('x') as stream:
            for position,entry in enumerate(entries):
                checked();row=entry['original_entry']
                prefix=tokenizer.encode('User: '+row['input']+'\nAssistant:',add_special_tokens=False).ids
                response=tokenizer.encode(row['output'],add_special_tokens=False).ids;ids=[1]+prefix+response+[2]
                native=len(ids)<=4096 and all(4<=v<32768 for v in ids[1:-1])
                base=dict(original_shard_position=position,source_id=row['id'],
                    original_entry_sha256=__import__('hashlib').sha256(json.dumps(row,sort_keys=True,ensure_ascii=True).encode()).hexdigest(),
                    complete_context_tokens=len(ids),response_EOS_targets=len(response)+1,complete_native_4096_fit=native,
                    protected_overlap=hits[str(position)],teacher_data=True,human_preferences=False,own_model_on_policy=False,
                    publisher_status=row['tests_execution_status'],publisher_score=row['average_test_score'],
                    optimization_admitted=False,independent_semantic_correctness_not_certified=True)
                try:parsed=parse_whole(row)
                except UnsupportedExample as error:
                    result=dict(base,status='unsupported_whole_example_abstention',reason=str(error));counts[result['status']]+=1
                    if position<1016 and previous[position]!=result:raise ValueError('Whole original unsupported disposition differs')
                else:
                    if position<1016:
                        result=previous[position]
                        if any(result[k]!=v for k,v in base.items()) or result['complete_original_tests']!=len(parsed['cases']):
                            raise ValueError('Every original whole-source identity/native/test field must reconstruct')
                        actual=result['actual_execution']
                        if actual['all_cases_observed']:
                            values=[ast.literal_eval(x) for x in actual['actual_value_literals']]
                            golds=[ast.literal_eval(x['expected_literal']) for x in parsed['cases']]
                            flags=[a==b for a,b in zip(values,golds)]
                            if len(values)!=len(golds) or flags!=actual['test_passes'] or all(flags)!=actual['all_published_tests_passed']:
                                raise ValueError('Every actual original raw program result must reconstruct')
                    else:
                        actual=observe(parsed,seconds=3);new_runs+=1
                        result=dict(base,actual_execution=actual,complete_original_tests=len(parsed['cases']))
                    eligible=actual['all_published_tests_passed'] and native and not hits[str(position)]
                    status='screen_pass_candidate' if eligible else 'not_screen_pass_candidate'
                    if position<1016 and (result['status']!=status or result['screen_pass_candidate']!=eligible):
                        raise ValueError('Original complete screening decision differs')
                    result.update(status=status,screen_pass_candidate=eligible)
                    counts[actual['status']]+=1
                    if eligible:counts['screen_pass_candidate']+=1;targets+=len(response)+1;contexts+=len(ids)
                counts['original_entries']+=1;counts['protected_overlap_entries']+=bool(hits[str(position)])
                counts['overlength_or_special_token_abstentions']+=not native
                stream.write(lines[position] if position<1016 else json.dumps(result,ensure_ascii=True)+'\n')
                if (position+1)%64==0:
                    atomic(OUT/'progress.json',dict(complete=position+1,actual_counts=dict(counts),fresh_tail_program_runs=new_runs,training_admitted=False))
        checked()
        if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every pinned original source must remain exact')
        atomic(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            all_original_dispositions_sha256=sha(OUT/'all1024-original-dispositions.jsonl'),actual_counts=dict(counts),
            screen_pass_response_EOS_targets=targets,screen_pass_context_tokens=contexts,complete_protected_input_stems=31349,
            previous_complete_dispositions_reconstructed_and_copied_byte_exact=1016,missing_original_source_rows_finished=8,
            fresh_tail_program_runs=new_runs,original_failed_producer_and_logs_preserved=True,
            all_unsupported_tests_abstain_whole_example=True,no_published_test_removed_or_repaired=True,
            published_teacher_tests_not_independent_semantic_oracles=True,every_original_row_retained=True,
            no_training_role_assigned=True,training_admitted=False,new_neural_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=True,actual_counts=dict(counts),new_tail_program_runs=new_runs,training_admitted=False)))
    except BaseException as error:
        atomic(OUT/'failure.json',dict(error=repr(error),training_admitted=False,goal_achieved=False));raise


if __name__=='__main__':main()
