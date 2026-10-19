"""Actual source-batch parity against established image and word encoders.

No neural model is loaded. These are representative tensor/interface controls,
not all-image pixel validation, complete training or a capability benchmark.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import resource

import torch

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.codenet_replay_v1 import FullReplayEncoder
from slm_lab.corrected_multimodal_batches_v1 import CorrectedMultimodalBatches
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.model import ModelConfig
from slm_lab.shared_word_data_v1 import SharedWordData

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/corrected-multimodal-batches-CPU-v1'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):value.update(block)
    return value.hexdigest()


def atomic(path,value):
    pending = path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh bounded one-thread low-priority CUDA-hidden source-batch audit required')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0));torch.set_num_threads(1)
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    config = ModelConfig(**read(ROOT/'data/public-reasoning-curriculum-v1/model-config.json'))
    source = CorrectedMultimodalBatches(ROOT,config,purpose='audit')
    if source.training_admitted:raise ValueError('This representative audit must not infer full source training admission')
    files = dict(source.files)
    files.update({str(p):sha(p) for p in python_sources(ROOT,[Path(__file__)])})
    def checked():
        wait_for_reserve(ROOT,minimum_bytes=40*1024**3)
        values = {k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
        used = values['anon']+values['shmem']+values['kernel'];limit = int(Path('/sys/fs/cgroup/memory.max').read_text())
        resident = next(int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
        if limit-used < 64*1024**3 or resident > 4*1024**3:raise ValueError('Bounded audit working memory and original other-process headroom required')
    checked();OUT.mkdir();owner = dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    cases = [(name,sorted({0,len(source.positions[name])-1,max(range(len(source.lengths[name])),key=source.lengths[name].__getitem__)})) for name in source.names]
    corrected = [item['original_prepared_native_view']['original_local_position'] for item in source.corrections]
    cases.append(('replay:image_reading',corrected))
    schedule = read(source.folder/'complete-corrected-single-epoch-schedule.json')['entries']
    photo = next(row for row in schedule if row['family'] == 'images:additional_photo_captions')
    cases.append((photo['family'],photo['original_local_positions']))
    longest = (0,0,0)
    with (ROOT/'data/shared-byte-words-v1/train.jsonl').open() as stream:
        for position,line in enumerate(stream):
            row = json.loads(line);longest = max(longest,(len(row['response'].encode()),row['minimum_CTC_steps'],position))
    cases.append(('words',sorted({0,21055,50000,68030,len(source.positions['words'])-1,longest[2]})))
    atomic(OUT/'plan.json',dict(owner=owner,source_files_sha256=files,actual_source_batch_cases=[dict(family=n,positions=p) for n,p in cases],
        actual_longest_native_context=max(max(v) for v in source.lengths.values()),actual_longest_word_UTF8_case=list(longest),
        every_source_family_included=True,canonical_image_encoder='FullReplayEncoder',canonical_word_encoder='SharedWordData',
        representative_batch_parity_not_full_physical_pixel_admission=True,scheduler='SCHED_IDLE',
        CUDA_hidden=True,no_neural_model_loaded=True,training_admitted=False,goal_achieved=False))
    try:
        golden = FullReplayEncoder(source.tokenizer,config,ROOT,448)
        words = SharedWordData(ROOT/'data/shared-byte-words-v1','train','audit')
        observations = [];actual_sources = native_targets = maximum_seen = 0
        for name,positions in cases:
            checked();actual = source.batch(name,positions)
            if name == 'words':
                expected = words.batch(positions,'cpu');canonical = {k:expected[k] for k in ['input_ids','labels','attention_mask','pixel_values']}
                if (not torch.equal(actual['word']['word_pixels'],expected['word_pixels']) or actual['word']['texts'] != expected['texts']
                        or actual['word']['native_byte_targets'] != [list(text.encode()) for text in expected['texts']]):
                    raise ValueError('Whole word pixels/transcripts/UTF8 targets must match the established original reader exactly')
            else:
                current_rows = [source.row(name,p)['current_row'] for p in positions]
                canonical = golden.batch(current_rows,'response','cpu')
            if set(actual['native']) != set(canonical) or any(not torch.equal(actual['native'][k],canonical[k]) for k in canonical):
                raise ValueError('Every actual native token/mask/target/pixel tensor must be bit exact to the established encoder')
            if actual['purpose'] != 'audit' or actual['training_admitted'] or actual['source_role'] != 'train':raise ValueError('Data roles and unadmitted audit status must remain exact')
            if name == 'images:additional_photo_captions':
                match = actual['matching']
                if match is None or len(match['captions']) != len(positions):raise ValueError('Every real photo needs its complete original matching caption and pixel key')
                for i,text in enumerate(match['captions']):
                    ids = [1]+source.tokenizer.encode(text,add_special_tokens=False).ids+[2]
                    if (match['text_ids'][i][match['text_mask'][i]].tolist() != ids
                            or match['pixel_keys'][i] != actual['source_observations'][i]['native_RGB448_sha256']):
                        raise ValueError('Caption-only complete matching inputs and exact visual positives must reproduce')
            elif actual['matching'] is not None:raise ValueError('No non-photo family may fabricate a paired caption objective')
            for row in actual['source_observations']:maximum_seen = max(maximum_seen,row['complete_context_tokens'])
            observations.append(dict(family=name,positions=positions,sources=len(positions),
                complete_context_tokens=actual['complete_context_tokens'],response_EOS_targets=actual['response_EOS_targets'],
                native_tensor_shapes={k:list(v.shape) for k,v in actual['native'].items()},
                actual_corrected_sources=[r['source_id'] for r in actual['source_observations'] if r['correction_index'] >= 0],
                canonical_native_tensors_bitexact=True,whole_word_pixels_and_bytes_bitexact=name == 'words',
                full_photo_caption_matching_inputs_rechecked=name == 'images:additional_photo_captions'))
            actual_sources += len(positions);native_targets += actual['response_EOS_targets']
            del actual,canonical
            if name == 'words':del expected
            atomic(OUT/'progress.json',dict(owner=owner,actual_complete_cases=len(observations),total_cases=len(cases),
                actual_native_source_presentations=actual_sources,training_admitted=False))
        if maximum_seen != 2952:raise ValueError('The actual longest complete native context must pass without clipping')
        for item in source.corrections:
            native = item['original_prepared_native_view'];negative = source.negative(native['family'],native['original_local_position'])
            original = source.row(native['family'],native['original_local_position'])['original_row']
            ids,labels = golden.sequence(original,'response')
            if negative is None or negative['native']['input_ids'] != ids or negative['native']['labels'] != labels:
                raise ValueError('Every complete original negative target and genuine EOS must remain retrievable intact')
        if source.negative(source.names[0],0) is not None:raise ValueError('No unqualified preference negative may be fabricated')
        # Actual singleton word-tail shards: only the owning rank may read a
        # source. Empty ranks must return without any row or image access.
        tail = schedule[3204]
        if tail['family'] != 'words' or tail['original_local_positions'] != [21055]:raise ValueError('Exact original singleton word tail required')
        owning = source.batch('words',[21055]);del owning
        original_row_method = source.row
        def no_read(*args,**kwargs):raise ValueError('An empty rank fabricated a source read')
        source.row = no_read
        for rank in [1,2,3]:
            if source.batch('words',tail['original_local_positions'][rank::4]) is not None:raise ValueError('Empty tail ranks must not fabricate a batch')
        source.row = original_row_method
        rejected = []
        for purpose in ['capacity','optimization']:
            try:CorrectedMultimodalBatches(ROOT,config,purpose=purpose)
            except ValueError as error:
                if 'Full physical-pixel and batch-source admission' not in str(error):raise
                rejected.append(purpose)
            else:raise ValueError('No capacity/optimization use before complete source admission is allowed')
        for name,positions in [('unknown',[0]),('words',[-1]),('words',[0,0]),('words',[len(source.positions['words'])])]:
            try:source.batch(name,positions)
            except ValueError:pass
            else:raise ValueError('Invalid original source requests must fail')
        try:source.batch('words',[0],device='cuda:0')
        except ValueError:pass
        else:raise ValueError('An audit request must not initialize a GPU batch')
        if torch.cuda.is_initialized():raise ValueError('No GPU context may be created by the source-only audit')
        if any(sha(p) != expected for p,expected in files.items()):raise ValueError('Every original/current source and adapter/runtime byte must remain unchanged')
        atomic(OUT/'complete.json',dict(complete=True,passed=True,at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
            actual_cases=observations,actual_source_families=len({n for n,_ in cases}),actual_complete_cases=len(cases),
            actual_native_source_presentations=actual_sources,actual_response_EOS_targets_checked=native_targets,
            actual_longest_native_context=maximum_seen,actual_longest_word_UTF8_case=list(longest),
            all_selected_native_tensors_bitexact_to_established_encoders=True,
            all4complete_original_preference_negatives_retrievable=True,
            singleton_word_tail_keeps_source_and_three_empty_ranks_never_read=True,
            actual_rejected_unadmitted_purposes=rejected,audit_GPU_request_and_invalid_positions_rejected=True,
            no_neural_model_or_GPU_context_created=True,
            representative_batch_scope_not_all_physical_images_or_full_size_capacity_or_quality=True,
            training_admitted=False,new_production_optimizer_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=True,passed=True,actual_families=9,actual_cases=len(cases),
            actual_native_sources=actual_sources,longest_native_context=maximum_seen,training_admitted=False)),flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json',dict(owner=owner,error=repr(error),automatic_retry=False,training_admitted=False,goal_achieved=False));raise


if __name__ == '__main__':main()
