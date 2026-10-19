"""Screen the entire pinned100,000-row public Python instruction shard.

CUDA-hidden, single-thread and nice19. The source file is already on disk.
No training, teacher API, external weights or new GPU lease is involved.
The output remains weak synthetic SFT candidates; it is never an RL oracle.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.public_code_instruction_screen_v1 import parse_whole,observe,UnsupportedExample
from slm_lab.public_reasoning_protected_text_v1 import population,overlaps,source_paths
from slm_lab.public_synthetic_code_screen_policy_v1 import published_all_pass,body_fingerprint,supervision_scope
from slm_lab.held_text_overlap_v1 import normalize

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/public-code-instruction-candidate-v1'
OUT=ROOT/'reports/full-public-code-shard-qualification-CPU-v1'
COLLECTION=ROOT/'reports/public-code-instruction-collection-CPU-v1/complete.json'
READBACK=ROOT/'reports/public-code-instruction-readback-CPU-v1'
VETO=ROOT/'reports/public-code-instruction-precision-counterexample-CPU-v1'
TOKENIZER=ROOT/'data/public-reasoning-curriculum-v1/tokenizer.json'
TESTS=ROOT/'reports/public-synthetic-code-screen-policy-v1-tests.log'
COUNT=100000
RESERVE=40*1024**3
MAX_OUTPUT=1024**3
WORKING=16*1024**3
OTHER_HEADROOM=64*1024**3


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def row_sha(row):return hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=True).encode()).hexdigest()


def atomic(path,value):
    pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def memory():
    values={k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
    return values['anon']+values['shmem']+values['kernel'],int(Path('/sys/fs/cgroup/memory.max').read_text())


def main():
    active,limit=memory()
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists()
            or os.environ.get('PYTHONHASHSEED')!='0' or os.environ.get('TOKENIZERS_PARALLELISM')!='false'
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            or '5 passed' not in TESTS.read_text() or limit-active<WORKING+OTHER_HEADROOM
            or shutil.disk_usage(ROOT).free<RESERVE+MAX_OUTPUT):
        raise ValueError('Fresh tested bounded low-priority one-thread CPU source audit with original headroom required')
    collection=read(COLLECTION);pilot=read(READBACK/'complete.json');known=read(VETO/'complete.json');veto=read(VETO/'semantic-veto.json')
    if (not all(r.get('complete') for r in [collection,pilot,known]) or any(r['training_admitted'] for r in [collection,pilot,known])
            or not known['independent_semantic_veto_for_this_source'] or collection['actual_shard_rows']!=COUNT
            or veto['evidence_sha256']!=sha(VETO/'complete.json')):
        raise ValueError('Pinned public shard, full pilot readback and unchanged mathematical veto required')
    files={**collection['source_files_sha256'],**pilot['source_files_sha256'],**known['source_files_sha256']}
    files.update({str(ROOT/p):digest for p,digest in collection['artifact_sha256'].items()})
    paths=[COLLECTION,READBACK/'complete.json',READBACK/'all1024-readback-observations.jsonl',VETO/'complete.json',
        VETO/'semantic-veto.json',TOKENIZER,TESTS,*source_paths(ROOT),
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_public_synthetic_code_screen_policy_v1.py'])]
    files.update({str(p):sha(p) for p in paths})
    if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every immutable original/pilot/runtime byte must remain exact')
    metadata={p:(Path(p).stat().st_dev,Path(p).stat().st_ino,Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in files}
    waits=Counter()
    def source_guard():
        if any((Path(p).stat().st_dev,Path(p).stat().st_ino,Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=m for p,m in metadata.items()):
            raise ValueError('A fixed source changed during full-shard qualification')
    def checked():
        result=wait_for_reserve(ROOT,minimum_bytes=RESERVE,guard=source_guard)
        waits['failed_space_observations']+=result['failed_space_observations'];waits['seconds']+=result['waited_seconds']
        active,limit=memory()
        if limit-active<OTHER_HEADROOM:raise ValueError('The original other-process memory headroom must remain available')
        resident=next(int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
        if resident>WORKING:raise ValueError('The declared CPU source-audit resident-memory allowance was exceeded')
    me=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    OUT.mkdir()
    atomic(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=me,source_files_sha256=files,
        dataset='nvidia/OpenCodeInstruct',revision=collection['revision'],exact_already_collected_shard=collection['source_file'],
        publisher_declared_license=collection['publisher_declared_license'],attribution=collection['attribution'],
        exact_whole_shard_rows=COUNT,synthetic_response_and_test_generator=collection['published_response_generator'],
        all_original_positions_retained=True,provider_nonpass_is_exclusion_not_independent_failure_label=True,
        every_remaining_original_assertion_required=True,complete_native_context_limit=4096,
        complete_protected_input_stems=31349,known_bad_original_hash_and_AST_veto=True,
        pilot_execution_reuse_requires_exact_original_sha_and_passed_fresh_readback=True,
        bounded_atomic_writer_space_wait_maximum_seconds=1800,minimum_free_bytes=RESERVE,maximum_output_bytes=MAX_OUTPUT,
        CPU_working_allowance=WORKING,other_process_memory_headroom=OTHER_HEADROOM,
        parent_address_space_ceiling=24*1024**3,active_memory_before=active,cgroup_memory_limit=limit,
        full_scope_original_training_human_RL_and_nine_endpoints_unchanged=True,
        no_external_weights_or_teacher_API_or_GPU_lease=True,scope=supervision_scope(),
        new_production_optimizer_updates=0,goal_achieved=False))
    try:
        resource.setrlimit(resource.RLIMIT_AS,(24*1024**3,24*1024**3))
        checked();parquet=pq.ParquetFile(DATA/'train-00000-of-00050.parquet')
        rows=[]
        for batch in parquet.iter_batches(batch_size=256,use_threads=False):rows.extend(batch.to_pylist())
        if len(rows)!=COUNT:raise ValueError('The complete whole pinned shard must reconstruct')
        if row_sha(rows[veto['original_shard_position']])!=veto['original_entry_sha256']:
            raise ValueError('The known mathematically wrong whole source must reconstruct exactly')
        bad_ast=body_fingerprint(parse_whole(rows[veto['original_shard_position']])['program'])
        with (READBACK/'all1024-readback-observations.jsonl').open() as stream:pilot_rows=list(map(json.loads,stream))
        with (DATA/'first1024-original-entries.jsonl').open() as stream:prefix=list(map(json.loads,stream))
        if len(pilot_rows)!=1024 or rows[:1024]!=[r['original_entry'] for r in prefix]:
            raise ValueError('Every reused pilot source must equal the whole original shard row')
        atomic(OUT/'progress.json',dict(owner=me,phase='full original shard reconstructed; scanning all protected stems',
            original_rows_loaded=COUNT,provider_complete_all_pass_rows=sum(published_all_pass(r) for r in rows),
            source_domain_counts=dict(Counter(r.get('domain') for r in rows)),original_dispositions_complete=0,
            training_admitted=False,new_production_optimizer_updates=0))
        scanner=[];valid_source=set()
        for position,row in enumerate(rows):
            if all(isinstance(row.get(k),str) and bool(row[k]) for k in ['id','input','output']):
                valid_source.add(position)
                scanner.append(dict(id=str(position),prompt=row['input'],independent_answer=row['output'],
                    original_entry=dict(question=row['input'])))
        # The scanner's legacy independent_answer field contains synthetic
        # teacher text only. No independence claim follows from that field name.
        hits,configuration=overlaps(scanner,population(ROOT),checked=checked)
        atomic(OUT/'protected-overlap-configuration.json',configuration)
        del scanner
        tokenizer=Tokenizer.from_file(str(TOKENIZER));counts=Counter();target_tokens=0;contexts=0;new_runs=0;reused=0
        with (OUT/'all100000-original-dispositions.jsonl').open('x') as stream:
            for position,row in enumerate(rows):
                if position%128==0:checked()
                result=dict(original_shard_position=position,source_id=row.get('id'),original_entry_sha256=row_sha(row),
                    source_domain=row.get('domain'),scope=supervision_scope(),screen_pass_candidate=False)
                if position not in valid_source:result['status']='malformed_whole_source_abstention'
                elif not published_all_pass(row):result['status']='provider_not_complete_all_pass_exclusion'
                else:
                    result['protected_overlap']=hits[str(position)]
                    result['input_only_group_sha256']=hashlib.sha256(normalize(row['input']).encode()).hexdigest()
                    if hits[str(position)]:result['status']='protected_input_overlap_exclusion'
                    else:
                        head=tokenizer.encode('User: '+row['input']+'\nAssistant:',add_special_tokens=False).ids
                        response=tokenizer.encode(row['output'],add_special_tokens=False).ids;ids=[1]+head+response+[2]
                        result.update(complete_context_tokens=len(ids),response_EOS_targets=len(response)+1)
                        if len(ids)>4096 or any(not 4<=v<32768 for v in ids[1:-1]):result['status']='whole_native_length_or_special_token_abstention'
                        else:
                            try:parsed=parse_whole(row)
                            except UnsupportedExample as error:result.update(status='unsupported_whole_test_set_abstention',reason=str(error))
                            else:
                                result['complete_original_tests']=len(parsed['cases']);fingerprint=body_fingerprint(parsed['program'])
                                result['program_AST_without_docstrings_sha256']=fingerprint
                                if row_sha(row)==veto['original_entry_sha256'] or fingerprint==bad_ast:
                                    result.update(status='known_independent_semantic_veto',veto_evidence=str(VETO/'complete.json'))
                                else:
                                    old=pilot_rows[position] if position<1024 else {}
                                    if old.get('fresh_whole_program_execution',{}).get('all_published_tests_passed'):
                                        observed=old['fresh_whole_program_execution'];reused+=1;result['execution_reused_from_exact_fresh_pilot_readback']=True
                                    else:
                                        observed=observe(parsed,seconds=3);new_runs+=1;result['execution_reused_from_exact_fresh_pilot_readback']=False
                                    result['actual_whole_execution']=observed
                                    if observed['all_published_tests_passed']:
                                        result.update(status='weak_SFT_screen_pass_candidate',screen_pass_candidate=True)
                                        target_tokens+=len(response)+1;contexts+=len(ids)
                                    else:result['status']=observed['status']
                counts[result['status']]+=1;counts['original_positions']+=1
                encoded=json.dumps(result,ensure_ascii=True)+'\n'
                if stream.tell()+len(encoded.encode())>MAX_OUTPUT:raise ValueError('The declared whole artifact storage ceiling was reached')
                stream.write(encoded)
                if (position+1)%128==0:
                    stream.flush()
                    progress=dict(owner=me,phase='whole source execution and screening',original_rows_loaded=COUNT,
                        original_dispositions_complete=position+1,actual_counts=dict(counts),actual_new_program_runs=new_runs,
                        actual_pinned_pilot_runs_reused=reused,candidate_response_EOS_targets=target_tokens,
                        bounded_disk_waits=dict(waits),training_admitted=False,new_production_optimizer_updates=0)
                    atomic(OUT/'progress.json',progress)
            stream.flush();os.fsync(stream.fileno())
        checked()
        if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every whole pinned original/pilot/runtime source must stay unchanged')
        atomic(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),owner=me,source_files_sha256=files,
            actual_original_positions=COUNT,actual_counts=dict(counts),actual_new_program_runs=new_runs,actual_pinned_pilot_runs_reused=reused,
            candidate_complete_context_tokens=contexts,candidate_response_EOS_targets=target_tokens,
            complete_protected_input_stems=31349,source_domain_counts=dict(Counter(r.get('domain') for r in rows)),
            all_original_dispositions_sha256=sha(OUT/'all100000-original-dispositions.jsonl'),bounded_disk_waits=dict(waits),
            all_original_shard_rows_retained_by_pinned_whole_parquet=True,whole_passing_synthetic_tests_not_independent_semantic_proof=True,
            fresh_full_readback_and_role_assignment_still_required=True,scope=supervision_scope(),training_admitted=False,
            source_model_weights_or_teacher_API_used=False,production_model_modified=False,new_production_optimizer_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=True,actual_counts=dict(counts),training_admitted=False)),flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json',dict(error=repr(error),owner=me,automatic_retry=False,training_admitted=False,goal_achieved=False));raise


if __name__=='__main__':main()
