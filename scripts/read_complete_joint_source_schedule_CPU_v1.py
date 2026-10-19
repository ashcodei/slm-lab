"""Fresh complete source/mask/schedule readback and human microbatch audit.

Reuse original independently admitted source corpora, freshly read every
declared train occurrence, then verify every combined reference. This checks
metadata and full native text/word targets; it does not re-decode all pixels,
run neural forwards, certify GPU speed or admit joint production training.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/complete-joint-source-schedule-CPU-v1'
OUT = ROOT / 'reports/complete-joint-source-schedule-readback-CPU-v1'
PRODUCER = dict(pid=277729, ticks='59627284')
RESERVE = 40 * 1024**3


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    if len(raw) > 16 * 1024**2 or shutil.disk_usage(ROOT).free - len(raw) < RESERVE:
        raise ValueError('The complete independent metadata must fit above the original writer reserve')
    pending = Path(path).with_name(Path(path).name + '.pending')
    with pending.open('xb') as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    pending.replace(path)


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])
            or shutil.disk_usage(ROOT).free < RESERVE + 32 * 1024**2):
        raise ValueError('Fresh one-thread CUDA-hidden nice19 metadata readback above40GiB is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.local_source_closure_v2 import python_sources
    if (read(SOURCE/'plan.json')['owner'] != PRODUCER or not terminal(PRODUCER)
            or (SOURCE/'failure.json').exists() or not read(SOURCE/'complete.json')['complete']):
        raise ValueError('The exact source producer must end successfully before independent readback')
    if '12 passed' not in (ROOT/'reports/joint-batched-human-v1-final-tests.log').read_text():
        raise ValueError('The actual new whole-pair gradient and joint neural integration checks must pass first')
    files = dict(read(SOURCE/'complete.json')['source_files_sha256'])
    seeds = [Path(__file__), ROOT/'src/slm_lab/joint_batched_human_preferences_v1.py',
        ROOT/'tests/test_joint_batched_human_preferences_v1.py', ROOT/'tests/test_complete_joint_source_schedule_v1.py']
    for p in [*python_sources(ROOT, seeds), *[SOURCE/name for name in ['plan.json','complete.json','joint.json','code.json','visual.json','human.json']],
            ROOT/'reports/joint-batched-human-v1-final-tests.log']:
        if str(p) in files and files[str(p)] != sha(p): raise ValueError('Source closures disagree')
        files[str(p)] = sha(p)
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    OUT.mkdir(); write(OUT/'plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        exact_completed_producer=PRODUCER, full_original_joint_updates=5362,
        source_only=True, GPU_contexts=0, models_constructed=0, goal_achieved=False))
    try:
        import numpy as np
        import torch
        from slm_lab.codenet_checkpoint_v1 import FileBindings
        from slm_lab.dense_human_preference_v1 import pair_information
        from slm_lab.joint_batched_human_preferences_v1 import plan
        from slm_lab.mixed_coding_SFT_data_v1 import MixedCodingSFTData
        from slm_lab.whole_human_data_v2 import WholeHumanData
        from slm_lab.full_visual_source_admission_v1 import FAMILIES
        torch.set_num_threads(1); guard = FileBindings(files); guard.check(full=True)
        parts = {name: read(SOURCE/(name+'.json')) for name in ['code','visual','human']}; counts = Counter()
        code = MixedCodingSFTData(ROOT); source_step = 0
        write(OUT/'progress.json', dict(owner=owner, phase='fresh complete code native inputs and three original epochs'))
        for epoch in range(3):
            order = list(range(28028)); random.Random(989923 + epoch).shuffle(order)
            for start in range(0, len(order), 16):
                positions = order[start:start+16]; info = [code.information(p) for p in positions]
                for p, original in zip(positions, info):
                    ids, labels = code.sequence(p); boundary = original['original_prompt_positions']
                    if (ids[0] != 1 or ids[-1] != 2 or labels != [-100]*boundary + ids[boundary:]
                            or len(ids) != original['complete_context_tokens'] or len(ids)-boundary != original['response_EOS_targets']
                            or hashlib.sha256(struct.pack('<'+'H'*len(ids), *ids)).hexdigest() != original['complete_native_input_sha256']):
                        raise ValueError('A complete original code sequence, prompt or EOS target differs')
                weights = [code.weight('balanced_sources', p) for p in positions]
                expected = dict(source_step=source_step+1, epoch=epoch, family='complete_mixed_code', original_positions=positions,
                    complete_context_tokens=sum(r['complete_context_tokens'] for r in info), response_EOS_targets=sum(r['response_EOS_targets'] for r in info),
                    rank_source_counts=[len(positions[r::4]) for r in range(4)], original_complete_source_information_sha256=digest(info),
                    original_balanced_source_weights=weights,
                    original_global_positive_weighted_targets=math.fsum(r['response_EOS_targets']*w for r,w in zip(info,weights)))
                if parts['code']['entries'][source_step] != expected: raise ValueError('An original complete code batch or denominator changed')
                source_step += 1; counts['complete_code_source_presentations'] += len(positions)
        if source_step != 5256 or len(parts['code']['entries']) != source_step: raise ValueError('The full original code horizon is required')
        write(OUT/'progress.json', dict(owner=owner, phase='fresh complete original visual index, words and four negatives', counts=dict(counts)))
        folder = ROOT/'reports/full-multimodal-corrected-source-index-CPU-v1'
        original = read(folder/'complete-corrected-single-epoch-schedule.json'); proof = read(folder/'complete.json')
        index = np.load(folder/'complete-native-source-index.npy', mmap_mode='r', allow_pickle=False)
        by_family = {name:index[index['family']==i] for i,name in enumerate(proof['family_names'])}
        word_bytes = []; tokens = np.memmap(ROOT/'data/shared-byte-words-v1/train.tokens.u16', dtype='<u2', mode='r')
        with (ROOT/'data/shared-byte-words-v1/train.jsonl').open('rb') as stream:
            for position, raw in enumerate(stream):
                row = json.loads(raw); item=row['native']; seq=tokens[item['offset']:item['offset']+item['length']]
                reference = by_family['words'][position]
                if (hashlib.sha256(raw).digest()!=reference['source_sha256'].tobytes()
                        or hashlib.sha256(seq.tobytes()).digest()!=reference['input_ids_sha256'].tobytes()
                        or len(seq)!=int(reference['context_tokens']) or int(seq[0])!=1 or int(seq[-1])!=2
                        or len(seq)-item['target_start']!=int(reference['response_EOS_targets'])):
                    raise ValueError('Every original full word, native token and EOS target must freshly reconstruct')
                word_bytes.append(len(row['response'].encode('utf8')))
        if len(word_bytes)!=91585 or sum(word_bytes)!=644063: raise ValueError('All91585whole words and644063UTF8bytes are required')
        corrections = [json.loads(raw) for raw in (ROOT/'data/public-document-visible-corrections-admitted-v1/all4-complete-corrections-and-preferences.jsonl').open()]
        occurrences = Counter(); negatives = 0
        for offset,row in enumerate(original['entries']):
            name=row['family']; positions=row['original_local_positions']; entries=by_family[name][positions]
            current=parts['visual']['entries'][offset]; selected=[int(n) for n in entries['correction_index'] if int(n)>=0]
            total=0
            for n in selected:
                negative=corrections[n]['complete_original_negative_native']; ids=negative['input_ids']; labels=negative['labels']; begin=negative['target_start']
                if ids[-1]!=2 or labels!=[-100]*begin+ids[begin:]: raise ValueError('Every complete original corrective negative must remain')
                total+=sum(v!=-100 for v in labels[1:])
            expected=dict(source_step=offset+1,epoch=0,family=name,original_positions=positions,
                complete_context_tokens=int(entries['context_tokens'].sum()),response_EOS_targets=int(entries['response_EOS_targets'].sum()),
                rank_source_counts=[len(positions[r::4]) for r in range(4)],whole_word_UTF8_bytes=sum(word_bytes[p] for p in positions) if name=='words' else 0,
                complete_original_negative_response_EOS_targets=total,complete_original_correction_indices=selected,original_corrected_entry_sha256=digest(row))
            if current!=expected: raise ValueError('A complete original visual entry or word/negative denominator changed')
            occurrences.update((name,p) for p in positions);negatives+=total
        if (occurrences!=Counter({(name,p):1 for name,n in FAMILIES.items() for p in range(n)})
                or len(parts['visual']['entries'])!=4733 or negatives!=80): raise ValueError('Every original visual occurrence and80negative targets are required')
        counts.update(complete_visual_views=302627,complete_word_views=91585,complete_word_UTF8_bytes=644063,retained_negative_targets=80)
        write(OUT/'progress.json', dict(owner=owner,phase='fresh whole human pairs and complete batched DPO call/padding audit',counts=dict(counts)))
        data=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','train','optimization')
        order=list(range(85786));random.Random(959117).shuffle(order)
        model_calls=source_tokens=padded_tokens=maximum_padded=maximum_batch=0
        for offset,start in enumerate(range(0,len(order),16)):
            positions=order[start:start+16];info=[pair_information(data,p) for p in positions]
            chosen=rejected=cc=rc=0
            for p in positions:
                a,al=data.sequence(p,'chosen');b,bl=data.sequence(p,'rejected');begin=data.rows[p]['native']['chosen']['target_start']
                if (a[:begin]!=b[:begin] or a[-1]!=2 or b[-1]!=2 or al!=[-100]*begin+a[begin:] or bl!=[-100]*begin+b[begin:]):
                    raise ValueError('Every complete original human prompt/response/EOS pair must freshly reconstruct')
                chosen+=sum(v!=-100 for v in al[1:]);rejected+=sum(v!=-100 for v in bl[1:]);cc+=len(a);rc+=len(b)
            expected=dict(source_step=offset+1,epoch=0,family='human_preferences',original_positions=positions,
                rank_source_counts=[len(positions[r::4]) for r in range(4)],complete_context_tokens=cc+rc,response_EOS_targets=chosen+rejected,
                chosen_context_tokens=cc,rejected_context_tokens=rc,chosen_response_EOS_targets=chosen,rejected_response_EOS_targets=rejected,
                complete_original_pair_information_sha256=digest(info))
            if parts['human']['entries'][offset]!=expected: raise ValueError('A complete original human source or normalization budget changed')
            for rank in range(4):
                actual,batches,padded=plan(data,positions,rank=rank,world=4,DPO_weight=1,maximum_examples=4,maximum_padded_tokens=16384)
                if actual!=info or sorted(p for batch in batches for p in batch)!=sorted(positions[rank::4]):
                    raise ValueError('Actual bounded human microbatches must preserve every original rank source')
                model_calls+=4*len(batches);padded_tokens+=sum(padded)
                maximum_padded=max(maximum_padded,max(padded,default=0));maximum_batch=max(maximum_batch,max(map(len,batches),default=0))
            source_tokens+=cc+rc;counts['complete_human_pairs']+=len(positions)
            if offset%500==0:write(OUT/'progress.json',dict(owner=owner,phase='complete human source and bounded DPO microbatch readback',human_batches=offset+1,counts=dict(counts)))
        if len(parts['human']['entries'])!=5362 or counts['complete_human_pairs']!=85786:raise ValueError('Every original human pair and full horizon is required')
        joint=read(SOURCE/'joint.json');used=Counter();combinations=Counter()
        for number,row in enumerate(joint['entries'],1):
            expected=[]
            for name in ['code','visual','human']:
                if number<=len(parts[name]['entries']):expected.append(dict(stream=name,source_step=number,original_entry_sha256=digest(parts[name]['entries'][number-1])))
            if row!=dict(step=number,components=expected):raise ValueError('Every actual complete joint reference must match its original source step')
            used.update(part['stream'] for part in expected);combinations['+'.join(part['stream'] for part in expected)]+=1
        if len(joint['entries'])!=5362 or used!=Counter(code=5256,visual=4733,human=5362) or combinations!=Counter({'code+visual+human':4733,'code+human':523,'human':106}):
            raise ValueError('No original full source step may be omitted or repeated')
        for name,part in parts.items():
            if digest(part)!=joint['component_streams_sha256'][name] or sha(SOURCE/(name+'.json'))!=joint['component_files'][name]['sha256']:
                raise ValueError('Every exact complete component identity must remain bound')
        guard.check(full=True)
        if torch.cuda.is_initialized(): raise ValueError('A source reader cannot acquire a CUDA context')
        write(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
            source_joint_sha256=sha(SOURCE/'joint.json'),actual_complete_joint_steps=5362,actual_original_counts=dict(counts),
            actual_human_microbatch_audit=dict(maximum_examples=4,maximum_padded_policy_tokens=16384,
                actual_model_calls_policy_plus_reference=model_calls,original_single_pair_policy_plus_reference_calls=4*85786,
                model_call_reduction_fraction=1-model_calls/(4*85786),original_policy_context_tokens=source_tokens,
                actual_padded_policy_context_slots=padded_tokens,padding_ratio=padded_tokens/source_tokens,
                largest_actual_microbatch_examples=maximum_batch,largest_actual_padded_policy_tokens=maximum_padded,
                actual_GPU_walltime_memory_and_speed_unmeasured=True),
            all_complete_original_source_masks_EOS_and_component_denominators_rechecked=True,
            full_original_visual_pixel_admission_reused_not_redecoded=True,
            joint_production_training_admitted=False,models_constructed=0,neural_forwards=0,optimizer_updates=0,goal_achieved=False))
    except BaseException as exc:
        write(OUT/'failure.json',dict(owner=owner,error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False,goal_achieved=False))
        raise


if __name__=='__main__':main()
