"""Fresh execution and input-overlap audit of every fixed public candidate.

No model is loaded, trained or rewarded. Both teacher programs and their
tests are synthetic. Actual test pass observations are screening evidence,
not independent semantic labels or an admission to model optimization.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

from tokenizers import Tokenizer

from slm_lab.public_code_instruction_screen_v1 import parse_whole, observe, UnsupportedExample
from slm_lab.public_reasoning_protected_text_v1 import population, overlaps, source_paths

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/public-code-instruction-candidate-v1'
OUT = ROOT/'reports/public-code-instruction-qualification-CPU-v1'
COLLECTION = ROOT/'reports/public-code-instruction-collection-CPU-v1/complete.json'
TESTS = ROOT/'reports/public-code-instruction-screen-v1-tests.log'
TOKENIZER = ROOT/'data/public-reasoning-curriculum-v1/tokenizer.json'
RESERVE = 40*1024**3


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def atomic(path,value):
    pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value,indent=2,ensure_ascii=True)+'\n')
        stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists()
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            or '5 passed' not in TESTS.read_text() or shutil.disk_usage(ROOT).free<RESERVE+128*1024**2):
        raise ValueError('Fresh tested bounded CPU-only source qualification required')
    collection=read(COLLECTION)
    if not collection.get('complete') or collection['training_admitted'] or collection['original_entries_retained']!=1024:
        raise ValueError('All fixed original candidate data must remain unadmitted')
    files=dict(collection['source_files_sha256'])
    files.update({str(ROOT/path):digest for path,digest in collection['artifact_sha256'].items()})
    paths=[Path(__file__),COLLECTION,TESTS,TOKENIZER,*source_paths(ROOT),
        ROOT/'src/slm_lab/public_code_instruction_screen_v1.py',
        ROOT/'scripts/public_code_literal_observations_v1.py',
        ROOT/'tests/test_public_code_instruction_screen_v1.py',
        ROOT/'src/slm_lab/public_reasoning_protected_text_v1.py',
        ROOT/'src/slm_lab/held_text_overlap_v1.py',ROOT/'src/slm_lab/standalone_program_v1.py',
        ROOT/'src/slm_lab/native_code.py',ROOT/'scripts/python_program_sandbox_v1.py',ROOT/'scripts/native_sandbox.py']
    files.update({str(path):sha(path) for path in paths})
    if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('Pinned original bytes differ')
    metadata={path:(Path(path).stat().st_dev,Path(path).stat().st_ino,Path(path).stat().st_size,Path(path).stat().st_mtime_ns) for path in files}
    def checked():
        if any((Path(p).stat().st_dev,Path(p).stat().st_ino,Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=original for p,original in metadata.items()):
            raise ValueError('Bound source changed during qualification')
        if shutil.disk_usage(ROOT).free<RESERVE:raise ValueError('Original writer reserve required')
    OUT.mkdir()
    atomic(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        all_original_entries=1024,maximum_seconds_per_program=3,complete_protected_input_stems=31349,
        whole_tests_only=True,unsupported_is_abstention=True,no_source_or_test_repair=True,
        whole_original_response_retained=True,publisher_flags_never_used_as_fresh_result=True,
        original_teacher_and_tests_synthetic=True,model_weights_or_teacher_API_used=False,
        training_admitted=False,new_neural_updates=0,goal_achieved=False))
    try:
        with (DATA/'first1024-original-entries.jsonl').open() as stream:entries=list(map(json.loads,stream))
        if len(entries)!=1024 or [r['original_shard_position'] for r in entries]!=list(range(1024)):
            raise ValueError('Every unchanged original prefix row required')
        # The scanner's legacy "independent_answer" field is only text input.
        # These values are explicitly synthetic teacher outputs, not independent answers.
        scanner=[dict(id=str(r['original_shard_position']),prompt=r['original_entry']['input'],
            independent_answer=r['original_entry']['output'],original_entry=dict(question=r['original_entry']['input'])) for r in entries]
        protected=population(ROOT)
        atomic(OUT/'progress.json',dict(phase='scanning every full question and response against all protected input stems',complete=0))
        hits,settings=overlaps(scanner,protected,checked=checked)
        atomic(OUT/'protected-overlap-configuration.json',settings)
        tokenizer=Tokenizer.from_file(str(TOKENIZER))
        counts=Counter();targets=0;contexts=0
        with (OUT/'all1024-original-dispositions.jsonl').open('x') as stream:
            for entry in entries:
                checked();position=entry['original_shard_position'];row=entry['original_entry']
                prefix=tokenizer.encode('User: '+row['input']+'\nAssistant:',add_special_tokens=False).ids
                response=tokenizer.encode(row['output'],add_special_tokens=False).ids
                ids=[1]+prefix+response+[2]
                complete_native=len(ids)<=4096 and all(4<=v<32768 for v in ids[1:-1])
                result=dict(original_shard_position=position,source_id=row['id'],
                    original_entry_sha256=hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=True).encode()).hexdigest(),
                    complete_context_tokens=len(ids),response_EOS_targets=len(response)+1,
                    complete_native_4096_fit=complete_native,protected_overlap=hits[str(position)],
                    teacher_data=True,human_preferences=False,own_model_on_policy=False,
                    publisher_status=row['tests_execution_status'],publisher_score=row['average_test_score'],
                    optimization_admitted=False,independent_semantic_correctness_not_certified=True)
                try:
                    parsed=parse_whole(row)
                except UnsupportedExample as error:
                    result.update(status='unsupported_whole_example_abstention',reason=str(error))
                    counts['unsupported_whole_example_abstention']+=1
                else:
                    result['actual_execution']=observe(parsed,seconds=3)
                    actual=result['actual_execution'];counts[actual['status']]+=1
                    result['complete_original_tests']=parsed['complete_original_tests']
                    eligible=actual['all_published_tests_passed'] and complete_native and not hits[str(position)]
                    result.update(status='screen_pass_candidate' if eligible else 'not_screen_pass_candidate',
                        screen_pass_candidate=eligible)
                    if eligible:
                        counts['screen_pass_candidate']+=1;targets+=len(response)+1;contexts+=len(ids)
                counts['original_entries']+=1
                counts['protected_overlap_entries']+=bool(hits[str(position)])
                counts['overlength_or_special_token_abstentions']+=not complete_native
                stream.write(json.dumps(result,ensure_ascii=True)+'\n');stream.flush()
                if (position+1)%16==0:
                    atomic(OUT/'progress.json',dict(phase='fresh complete-source executions',complete=position+1,
                        actual_counts=dict(counts),training_admitted=False,new_neural_updates=0))
        checked()
        if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('All bound inputs must remain bit exact')
        atomic(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            all_original_dispositions_sha256=sha(OUT/'all1024-original-dispositions.jsonl'),actual_counts=dict(counts),
            screen_pass_response_EOS_targets=targets,screen_pass_context_tokens=contexts,
            complete_protected_input_stems=31349,all_unsupported_tests_abstain_whole_example=True,
            no_published_test_removed_or_repaired=True,published_teacher_tests_not_independent_semantic_oracles=True,
            every_original_row_retained=True,no_training_role_assigned=True,
            training_admitted=False,new_neural_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=True,actual_counts=dict(counts),screen_pass_targets=targets,training_admitted=False)))
    except BaseException as error:
        atomic(OUT/'failure.json',dict(error=repr(error),training_admitted=False,goal_achieved=False))
        raise


if __name__=='__main__':main()
