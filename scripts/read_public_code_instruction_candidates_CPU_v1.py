"""Replay every source disposition and freshly execute every surviving candidate.

This is a candidate-data readback, not model training admission. A correct
answer to the natural-language question is not certified by synthetic tests.
"""
from collections import Counter
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from slm_lab.public_code_instruction_screen_v1 import parse_whole,observe,UnsupportedExample
from slm_lab.public_reasoning_protected_text_v1 import population,overlaps

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/public-code-instruction-candidate-v1'
QUAL=ROOT/'reports/public-code-instruction-qualification-CPU-v2'
OUT=ROOT/'reports/public-code-instruction-readback-CPU-v1'
TOKENIZER=ROOT/'data/public-reasoning-curriculum-v1/tokenizer.json'
VETO=ROOT/'reports/public-code-instruction-precision-counterexample-CPU-v1'
RESERVE=40*1024**3


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def atomic(path,value):
    pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:stream.write(json.dumps(value,indent=2,ensure_ascii=True)+'\n')
    pending.replace(path)


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists()
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            or shutil.disk_usage(ROOT).free<RESERVE+128*1024**2):
        raise ValueError('Fresh bounded low-priority single-thread CPU readback required')
    proof=read(QUAL/'complete.json')
    if (not proof.get('complete') or proof['training_admitted'] or proof['actual_counts']['original_entries']!=1024
            or proof['complete_protected_input_stems']!=31349):raise ValueError('Complete original unadmitted qualification required')
    files=dict(proof['source_files_sha256'])
    semantic=read(VETO/'complete.json');veto=read(VETO/'semantic-veto.json')
    if (not semantic.get('complete') or not semantic['independent_semantic_veto_for_this_source']
            or semantic['training_admitted'] or veto['evidence_sha256']!=sha(VETO/'complete.json')):
        raise ValueError('The actual independent mathematical counterexample must remain an explicit exclusion')
    files.update(semantic['source_files_sha256'])
    files.update({str(p):sha(p) for p in [Path(__file__),QUAL/'complete.json',QUAL/'all1024-original-dispositions.jsonl',QUAL/'protected-overlap-configuration.json',VETO/'complete.json',VETO/'semantic-veto.json']})
    if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every pinned original source and disposition required')
    metadata={p:(Path(p).stat().st_dev,Path(p).stat().st_ino,Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in files}
    def checked():
        if any((Path(p).stat().st_dev,Path(p).stat().st_ino,Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=m for p,m in metadata.items()):
            raise ValueError('Bound source changed during readback')
        if shutil.disk_usage(ROOT).free<RESERVE:raise ValueError('Original persistent writer reserve required')
    OUT.mkdir()
    atomic(OUT/'plan.json',dict(source_files_sha256=files,all_original_source_entries=1024,
        fresh_execution_of_every_screen_pass_candidate=True,known_independent_semantic_veto=veto,
        training_admitted=False,new_neural_updates=0,goal_achieved=False))
    try:
        with (DATA/'first1024-original-entries.jsonl').open() as stream:entries=list(map(json.loads,stream))
        with (QUAL/'all1024-original-dispositions.jsonl').open() as stream:dispositions=list(map(json.loads,stream))
        raw=[]
        for batch in pq.ParquetFile(DATA/'train-00000-of-00050.parquet').iter_batches(batch_size=128,use_threads=False):
            raw.extend(batch.to_pylist())
            if len(raw)>=1024:break
        if len(entries)!=1024 or len(dispositions)!=1024 or raw[:1024]!=[r['original_entry'] for r in entries]:
            raise ValueError('Complete original prefix must reconstruct directly from the whole pinned shard')
        veto_row=entries[veto['original_shard_position']]['original_entry']
        if (veto_row['id']!=veto['source_id']
                or hashlib.sha256(json.dumps(veto_row,sort_keys=True,ensure_ascii=True).encode()).hexdigest()!=veto['original_entry_sha256']):
            raise ValueError('The known incorrect original teacher answer must stay excluded by its whole original hash')
        case=semantic['independent_exact_integer_counterexample'];points=ast.literal_eval(case['input_literal'])
        distances=[((points[i][0]-points[j][0])**2+(points[i][1]-points[j][1])**2,(points[i],points[j]))
            for i in range(len(points)) for j in range(i+1,len(points))]
        best=min(distances,key=lambda r:r[0])
        if (not 2<=len(points)<=1000 or not all(type(v) is int and -10**9<=v<=10**9 for p in points for v in p)
                or sum(r[0]==best[0] for r in distances)!=1
                or best[1]!=ast.literal_eval(case['unique_exact_expected_pair_literal'])
                or best[1]==ast.literal_eval(case['actual_unchanged_teacher_pair_literal'])):
            raise ValueError('Original bounds, unique exact mathematical optimum and observed counterexample must reconstruct')
        scanner=[dict(id=str(r['original_shard_position']),prompt=r['original_entry']['input'],
            independent_answer=r['original_entry']['output'],original_entry=dict(question=r['original_entry']['input'])) for r in entries]
        hits,settings=overlaps(scanner,population(ROOT),checked=checked)
        if settings!=read(QUAL/'protected-overlap-configuration.json'):raise ValueError('Full protected scan settings differ')
        tokenizer=Tokenizer.from_file(str(TOKENIZER));counts=Counter();fresh=0;targets=0;contexts=0
        with (OUT/'all1024-readback-observations.jsonl').open('x') as stream:
            for position,(entry,prior) in enumerate(zip(entries,dispositions)):
                checked();row=entry['original_entry']
                if entry['original_shard_position']!=position or prior['original_shard_position']!=position or prior['source_id']!=row['id']:
                    raise ValueError('Every unchanged original identity and position required')
                if prior['original_entry_sha256']!=hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=True).encode()).hexdigest():
                    raise ValueError('Whole original row hash differs')
                prefix=tokenizer.encode('User: '+row['input']+'\nAssistant:',add_special_tokens=False).ids
                response=tokenizer.encode(row['output'],add_special_tokens=False).ids
                ids=[1]+prefix+response+[2]
                complete_native=len(ids)<=4096 and all(4<=v<32768 for v in ids[1:-1])
                if (prior['complete_context_tokens']!=len(ids) or prior['response_EOS_targets']!=len(response)+1
                        or prior['complete_native_4096_fit']!=complete_native or prior['protected_overlap']!=hits[str(position)]
                        or tokenizer.decode(prefix,skip_special_tokens=False)!='User: '+row['input']+'\nAssistant:'
                        or tokenizer.decode(response,skip_special_tokens=False)!=row['output']):
                    raise ValueError('Whole untruncated native framing and protected overlap must reconstruct')
                observation=dict(original_shard_position=position,original_source_and_native_masks_rebuilt=True,
                    known_independent_semantic_veto=position==veto['original_shard_position'])
                try:parsed=parse_whole(row)
                except UnsupportedExample as error:
                    if prior['status']!='unsupported_whole_example_abstention' or prior['reason']!=str(error):
                        raise ValueError('Unsupported whole source disposition must reconstruct')
                    counts['unsupported_whole_example_abstention']+=1
                else:
                    actual=prior['actual_execution'];counts[actual['status']]+=1
                    if prior['complete_original_tests']!=len(parsed['cases']):raise ValueError('Every original test must remain')
                    if actual['all_cases_observed']:
                        values=[ast.literal_eval(x) for x in actual['actual_value_literals']]
                        golds=[ast.literal_eval(x['expected_literal']) for x in parsed['cases']]
                        flags=[a==b for a,b in zip(values,golds)]
                        if len(values)!=len(golds) or flags!=actual['test_passes'] or all(flags)!=actual['all_published_tests_passed']:
                            raise ValueError('All published expectations must compare against the actual whole raw outputs')
                    eligible=actual['all_published_tests_passed'] and complete_native and not hits[str(position)]
                    if prior['screen_pass_candidate']!=eligible:raise ValueError('Original screen decision differs')
                    if eligible:
                        replay=observe(parsed,seconds=3);fresh+=1
                        observation['fresh_whole_program_execution']=replay
                        if not replay['all_published_tests_passed'] or replay['whole_original_tests']!=len(parsed['cases']):
                            raise ValueError('A previously passing candidate did not survive fresh complete execution')
                        counts['screen_pass_candidate']+=1;targets+=len(response)+1;contexts+=len(ids)
                counts['original_entries']+=1;counts['protected_overlap_entries']+=bool(hits[str(position)])
                counts['overlength_or_special_token_abstentions']+=not complete_native
                stream.write(json.dumps(observation,ensure_ascii=True)+'\n');stream.flush()
                if (position+1)%32==0:
                    atomic(OUT/'progress.json',dict(original_dispositions_rebuilt=position+1,fresh_candidate_program_runs=fresh,training_admitted=False))
        if (dict(counts)!=proof['actual_counts'] or fresh!=counts['screen_pass_candidate']
                or targets!=proof['screen_pass_response_EOS_targets'] or contexts!=proof['screen_pass_context_tokens']):
            raise ValueError('Every original aggregate must reconstruct')
        checked()
        if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every source byte must remain unchanged')
        atomic(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            actual_counts=dict(counts),original_entries_and_full_native_masks_rebuilt=1024,
            complete_protected_input_stems_rechecked=31349,fresh_complete_candidate_program_runs=fresh,
            screen_pass_response_EOS_targets=targets,screen_pass_context_tokens=contexts,
            known_screen_pass_but_mathematically_wrong_source=veto,
            independent_exact_integer_counterexample_rebuilt=True,
            all_screen_pass_sources_remain_unadmitted_pending_semantic_qualification=True,
            synthetic_teacher_and_tests=True,independent_semantic_correctness_not_certified=True,
            training_admitted=False,model_has_consumed_rows=False,new_neural_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=True,all_original_entries=1024,fresh_candidate_program_runs=fresh,training_admitted=False)))
    except BaseException as error:
        atomic(OUT/'failure.json',dict(error=repr(error),training_admitted=False,goal_achieved=False))
        raise


if __name__=='__main__':main()
