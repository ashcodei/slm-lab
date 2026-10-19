"""Bind every original word/native transcript and full word microbatch plan.

This CPU-only metadata/native readback does not decode pixels, perform neural
forwards or admit a new model-training stage. Every91585source occurrence,
644063UTF8bytes and436491native response/EOS targets must be retained.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import resource

import numpy as np
from tokenizers import Tokenizer

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.global_native_visual_stage_kernel_v1 import native_microbatches
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.multimodal_source_index_format_v1 import DTYPE,FORMAT
from slm_lab.shared_word_data_v1 import native_bytes

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/full-multimodal-corrected-source-index-CPU-v1'
READER=ROOT/'reports/full-multimodal-corrected-source-index-v1-readback/complete.json'
WORDS=ROOT/'data/shared-byte-words-v1/train.jsonl'
TOKENS=ROOT/'data/shared-byte-words-v1/train.tokens.u16'
TOKENIZER=ROOT/'data/tokenizer-corpus/tokenizer.json'
TESTS=ROOT/'reports/global-word-microbatch-stage-v1-tests.log'
OUT=ROOT/'reports/full-word-microbatch-source-audit-CPU-v1'
RESERVE=40*1024**3


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def atomic(path,value):
    pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            or '8 passed' not in TESTS.read_text()):
        raise ValueError('Fresh tested CUDA-hidden one-thread nice19 source-only audit required')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0));wait_for_reserve(ROOT,minimum_bytes=RESERVE)
    resource.setrlimit(resource.RLIMIT_AS,(24*1024**3,24*1024**3))
    proof,reader=read(SOURCE/'complete.json'),read(READER)
    index_path=SOURCE/'complete-native-source-index.npy';schedule_path=SOURCE/'complete-corrected-single-epoch-schedule.json'
    if (not proof['complete'] or not reader['complete'] or proof['format']!=FORMAT
            or proof['training_admitted'] or reader['training_admitted']
            or proof['actual_word_source_rows']!=91585 or proof['actual_whole_CTC_UTF8_byte_targets']!=644063
            or proof['index_sha256']!=sha(index_path) or proof['schedule_sha256']!=sha(schedule_path)):
        raise ValueError('Exact whole corrected source index/readback and unchanged full source schedule required')
    original={**proof['source_files_sha256'],**reader['source_files_sha256']}
    for path in [WORDS,TOKENS,TOKENIZER]:
        if str(path) not in original or sha(path)!=original[str(path)]:
            raise ValueError('Exact original whole word/source/tokenizer bytes must remain bound')
    files={str(p):sha(p) for p in [SOURCE/'complete.json',READER,index_path,schedule_path,WORDS,TOKENS,TOKENIZER,TESTS,
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_global_word_microbatch_stage_v1.py'])]}
    states={p:(Path(p).stat().st_dev,Path(p).stat().st_ino,Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in files}
    def guard():
        if os.sched_getscheduler(0)!=os.SCHED_IDLE or any(
                (Path(p).stat().st_dev,Path(p).stat().st_ino,Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=v for p,v in states.items()):
            raise ValueError('All original source bytes and idle scheduling must remain exact')
    def checked():wait_for_reserve(ROOT,minimum_bytes=RESERVE,guard=guard)
    me=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    OUT.mkdir();atomic(OUT/'plan.json',dict(owner=me,source_files_sha256=files,all_original_word_rows=91585,
        all_original_word_UTF8_bytes=644063,all_original_native_word_response_EOS_targets=436491,
        full_word_batches=1432,world=4,maximum_microbatch_examples=4,maximum_padded_decoder_tokens=4096,
        original_CTC_frames=140,original_CTC_blank=256,original_word_readout='FP32',
        whole_transcript_SHA_and_native_decode_rechecked=True,pixels_not_redecoded=True,
        neural_forwards=0,training_admitted=False,new_production_optimizer_updates=0,goal_achieved=False))
    try:
        all_index=np.load(index_path,mmap_mode='r',allow_pickle=False)
        if all_index.dtype!=DTYPE or all_index.shape!=(302627,):raise ValueError('Exact complete compact index schema required')
        family=proof['family_names'].index('words');index=all_index[all_index['family']==family]
        if len(index)!=91585 or not np.array_equal(index['original_local_position'],np.arange(91585)):
            raise ValueError('Every complete original ordered word occurrence required')
        tokens=np.memmap(TOKENS,dtype='<u2',mode='r');tokenizer=Tokenizer.from_file(str(TOKENIZER))
        if tokenizer.padding is not None or tokenizer.truncation is not None:raise ValueError('No hidden native clipping/padding allowed')
        identities=[];counts=Counter();families=Counter();longest=dict(UTF8_bytes=0)
        observations=OUT/'all91585-complete-word-native-identities.jsonl'
        with WORDS.open('rb') as stream,observations.open('x') as output:
            for position,entry in enumerate(index):
                if position%1024==0:checked()
                offset=stream.tell();line=stream.readline()
                if not line.endswith(b'\n'):raise ValueError('Every source row must retain its complete LF-terminated record')
                row=json.loads(line);native=row['native'];text=row['response'];values,required=native_bytes(text)
                sequence=tokens[native['offset']:native['offset']+native['length']].astype(int).tolist()
                start=native['target_start'];body=sequence[start:-1]
                if (row['split']!='train' or row['task']!='word_ocr' or row['native_byte_targets']!=values
                        or row['minimum_CTC_steps']!=required or int(entry['correction_index'])!=-1
                        or int(entry['source_offset'])!=offset or int(entry['source_bytes'])!=len(line)
                        or entry['source_sha256'].tobytes()!=hashlib.sha256(line).digest()
                        or native['length']!=int(entry['context_tokens']) or start!=int(entry['target_start'])
                        or len(sequence)!=native['length'] or sequence[0]!=1 or sequence[-1]!=2
                        or any(not 4<=v<32768 for v in sequence[1:-1])
                        or len(sequence)-start!=int(entry['response_EOS_targets'])
                        or hashlib.sha256(np.asarray(sequence,dtype='<u2').tobytes()).digest()!=entry['input_ids_sha256'].tobytes()
                        or tokenizer.decode(body,skip_special_tokens=False)!=text):
                    raise ValueError('Every whole original word/source/native byte range and genuine EOS response must reconstruct')
                identity=dict(original_local_position=position,source_id=row['id'],source_role='train',
                    source_row_offset=offset,source_row_bytes=len(line),source_row_sha256=hashlib.sha256(line).hexdigest(),
                    whole_word_UTF8_sha256=hashlib.sha256(bytes(values)).hexdigest(),whole_word_UTF8_bytes=len(values),
                    minimum_repeated_byte_CTC_steps=required,complete_context_tokens=len(sequence),target_start=start,
                    response_EOS_targets=len(sequence)-start,native_input_ids_sha256=entry['input_ids_sha256'].tobytes().hex(),
                    byte_family=row['byte_family'],training_admitted=False)
                identities.append(identity)
                counts['words']+=1;counts['UTF8_bytes']+=len(values);counts['context_tokens']+=len(sequence)
                counts['response_EOS_targets']+=len(sequence)-start;families[row['byte_family']]+=1
                if len(values)>longest['UTF8_bytes']:
                    longest=dict(original_local_position=position,source_id=row['id'],UTF8_bytes=len(values),minimum_CTC_steps=required)
                encoded=json.dumps(identity)+'\n'
                if output.tell()+len(encoded.encode())>128*1024**2:raise ValueError('The full identity output exceeded its storage allowance')
                output.write(encoded)
            if stream.read(1):raise ValueError('No original source row may remain outside the full audit')
            output.flush();os.fsync(output.fileno())
        if counts!=Counter(words=91585,UTF8_bytes=644063,context_tokens=1810266,response_EOS_targets=436491):
            raise ValueError('Every full original word/native exposure budget must remain unchanged')
        steps=0;microbatches=0;seen=set();tails=[];schedule=read(schedule_path)
        plans=OUT/'all1432-complete-word-rank-microbatch-plans.jsonl'
        with plans.open('x') as output:
            for entry in schedule['entries']:
                if entry['family']!='words':continue
                if steps%128==0:checked()
                positions=entry['original_local_positions'];rows=[identities[p] for p in positions];ranks=[]
                if any(p in seen for p in positions):raise ValueError('No original word occurrence may be repeated')
                for rank in range(4):
                    local=positions[rank::4];lengths=[identities[p]['complete_context_tokens'] for p in local]
                    chunks=native_microbatches(lengths,prefix_tokens=196);microbatches+=len(chunks)
                    rank_plan=[]
                    for chunk in chunks:
                        chosen=[identities[local[i]] for i in chunk]
                        rank_plan.append(dict(original_positions=[local[i] for i in chunk],
                            complete_context_lengths=[v['complete_context_tokens'] for v in chosen],
                            complete_UTF8_bytes=[v['whole_word_UTF8_bytes'] for v in chosen],
                            complete_UTF8_sha256=[v['whole_word_UTF8_sha256'] for v in chosen],
                            response_EOS_targets=sum(v['response_EOS_targets'] for v in chosen),
                            padded_decoder_tokens=len(chosen)*(max(v['complete_context_tokens'] for v in chosen)+196)))
                    if [p for v in rank_plan for p in v['original_positions']]!=local:
                        raise ValueError('Every whole rank word must reach its exact original microbatch once')
                    ranks.append(dict(rank=rank,original_positions=local,microbatches=rank_plan))
                if len(positions)<4:tails.append(dict(step=entry['step'],positions=positions,rank_counts=[len(v['original_positions']) for v in ranks]))
                if (sum(v['complete_context_tokens'] for v in rows)!=entry['complete_context_tokens']
                        or sum(v['response_EOS_targets'] for v in rows)!=entry['response_EOS_targets']):
                    raise ValueError('Every original full word batch context/target denominator must remain exact')
                output.write(json.dumps(dict(step=entry['step'],global_positions=positions,
                    global_word_UTF8_sha256=[v['whole_word_UTF8_sha256'] for v in rows],
                    global_word_UTF8_bytes=[v['whole_word_UTF8_bytes'] for v in rows],
                    global_context_lengths=[v['complete_context_tokens'] for v in rows],
                    global_response_EOS_targets=[v['response_EOS_targets'] for v in rows],ranks=ranks,
                    training_admitted=False))+'\n');seen.update(positions);steps+=1
                if output.tell()>128*1024**2:raise ValueError('The full word plan exceeded its storage allowance')
            output.flush();os.fsync(output.fileno())
        if seen!=set(range(91585)) or steps!=1432 or tails!=[dict(step=3205,positions=[21055],rank_counts=[1,0,0,0])]:
            raise ValueError('All91585word occurrences,1432updates and original singleton tail must remain intact')
        checked()
        if any(sha(p)!=value for p,value in files.items()):raise ValueError('Every bound original/runtime byte must remain unchanged')
        atomic(OUT/'complete.json',dict(complete=True,passed=True,at=datetime.now(timezone.utc).isoformat(),owner=me,
            source_files_sha256=files,actual_counts=dict(counts),actual_byte_families=dict(families),
            actual_word_global_batches=steps,actual_rank_microbatches=microbatches,actual_partial_word_tail=tails,
            actual_longest_UTF8_transcript=longest,all_whole_native_responses_decode_to_original_transcripts=True,
            all1432global_and_rank_whole_UTF8_hashes_bound=True,
            identities_sha256=sha(observations),full_word_microbatch_plans_sha256=sha(plans),
            original_CTC_blank=256,original_CTC_frames=140,original_FP32_word_readout_preserved=True,
            every_current_neural_worker_and_original_human_RL_and_nine_endpoints_unchanged=True,
            full_pixel_source_and_distributed_full_size_GPU_capacity_recovery_training_quality_pending=True,
            old_different_batch_shape_raw_gradient_gate_not_waived=True,
            neural_forwards=0,training_admitted=False,new_production_optimizer_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=True,passed=True,counts=dict(counts),word_batches=steps,microbatches=microbatches,longest=longest)),flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json',dict(owner=me,error=repr(error),automatic_retry=False,training_admitted=False));raise


if __name__=='__main__':main()
