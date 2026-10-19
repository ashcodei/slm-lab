"""Read every original source and publish a complete joint metadata recipe.

No model, optimizer, CUDA context, GPU lease, worker launch or live source
mutation. Full original independent experiments and future joint runtime,
capacity, storage, training/export and assessments remain separate work.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/complete-joint-source-schedule-CPU-v1'
RESERVE = 40 * 1024**3
MAX_OUTPUT = 128 * 1024**2


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    if len(raw) > 64 * 1024**2 or shutil.disk_usage(ROOT).free - len(raw) < RESERVE:
        raise ValueError('Complete source metadata must fit its declared bound above the original reserve')
    allocated = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    if allocated + len(raw) > MAX_OUTPUT: raise ValueError('Complete joint metadata exceeds its full physical output budget')
    pending = Path(path).with_name(Path(path).name + '.pending')
    with pending.open('xb') as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    pending.replace(path)


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])
            or shutil.disk_usage(ROOT).free < RESERVE + MAX_OUTPUT):
        raise ValueError('Fresh CUDA-hidden one-thread nice19 preparation needs the entire metadata budget above40GiB')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from slm_lab.complete_joint_source_schedule_v1 import combine, digest, require_original_scope
    from slm_lab.local_source_closure_v2 import python_sources
    entrypoints = [Path(__file__), ROOT / 'src/slm_lab/complete_joint_source_schedule_v1.py',
        ROOT / 'src/slm_lab/mixed_coding_SFT_data_v1.py', ROOT / 'src/slm_lab/whole_human_data_v2.py',
        ROOT / 'src/slm_lab/dense_human_preference_v1.py', ROOT / 'src/slm_lab/codenet_stage_v1.py']
    files = {str(p): sha(p) for p in python_sources(ROOT, entrypoints)}
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    OUT.mkdir()
    write(OUT / 'plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        whole_code_passes=3, whole_visual_passes=1, whole_human_passes=1, maximum_output_bytes=MAX_OUTPUT,
        minimum_other_writer_reserve_bytes=RESERVE, models_constructed=0, optimizer_updates=0, GPU_leases=0, goal_achieved=False))
    try:
        import numpy as np
        import torch
        from slm_lab.codenet_checkpoint_v1 import FileBindings
        from slm_lab.codenet_stage_v1 import EpochOrder
        from slm_lab.dense_human_preference_v1 import pair_information
        from slm_lab.mixed_coding_SFT_data_v1 import MixedCodingSFTData
        from slm_lab.whole_human_data_v2 import WholeHumanData
        torch.set_num_threads(1)
        write(OUT / 'progress.json', dict(owner=owner, phase='checking every original admitted mixed code source', models_constructed=0))
        code = MixedCodingSFTData(ROOT); original_code = code.complete_schedule(epochs=3, width=16, seed=989923)
        binding = read(ROOT / 'reports/mixed-coding-SFT-source-bindings-CPU-v1/complete.json')
        if code.identity_sha256 != binding['mixed_source_identity_sha256'] or code.budget(original_code) != binding['full_training_budget']:
            raise ValueError('Every actual original mixed source identity and complete three-epoch budget must match')
        for p, h in code.source_files_sha256.items():
            if p in files and files[p] != h: raise ValueError('Original source evidence disagrees')
            files[p] = h
        code_rows = []
        per_epoch = len(original_code) // 3
        for index, positions in enumerate(original_code):
            info = code.metadata(positions, 'balanced_sources')
            code_rows.append(dict(source_step=index + 1, epoch=index // per_epoch, family='complete_mixed_code', original_positions=positions,
                complete_context_tokens=info['complete_context_tokens'], response_EOS_targets=info['response_EOS_targets'],
                rank_source_counts=[len(positions[r::4]) for r in range(4)],
                original_complete_source_information_sha256=digest(info['complete_sources']),
                original_balanced_source_weights=info['actual_source_weights'],
                original_global_positive_weighted_targets=info['actual_global_weighted_targets']))
        if any(Counter(p for r in code_rows if r['epoch'] == e for p in r['original_positions']) != Counter(range(28028)) for e in range(3)):
            raise ValueError('Every complete original code source must occur once in every original epoch')
        streams = dict(code=dict(source_role='train', source_identity_sha256=code.identity(original_code), seed=989923,
            positive_family_weights=code.family_weights, public_synthetic_reward_oracle_forbidden=True, entries=code_rows))
        write(OUT / 'code.json', streams['code']); code.guard.check(full=True)
        write(OUT / 'progress.json', dict(owner=owner, phase='checking every original visual and whole-word source', code_presentations=84084))
        folder = ROOT / 'reports/full-multimodal-corrected-source-index-CPU-v1'
        visual_path = folder / 'complete-corrected-single-epoch-schedule.json'
        visual = read(visual_path); proof = read(folder / 'complete.json'); admission = read(folder / 'source-training-admission.json')
        if admission.get('training_admitted') is not True or admission.get('admitted_role') != 'train' or admission['index_sha256'] != proof['index_sha256']:
            raise ValueError('The separate original full visual source admission is required')
        for p, h in admission['source_files_sha256'].items():
            if p in files and files[p] != h: raise ValueError('Original visual source evidence disagrees')
            files[p] = h
        index_path = folder / 'complete-native-source-index.npy'
        if sha(index_path) != proof['index_sha256']: raise ValueError('The original complete visual index differs')
        index = np.load(index_path, mmap_mode='r', allow_pickle=False)
        family_rows = {name: index[index['family'] == number] for number, name in enumerate(proof['family_names'])}
        word_path = ROOT / 'data/shared-byte-words-v1/train.jsonl'; word_bytes = []
        with word_path.open() as word_stream:
            for position, raw in enumerate(word_stream):
                row = json.loads(raw); native = row['native']; idx = family_rows['words'][position]
                if (row['split'] != 'train' or native['length'] != int(idx['context_tokens'])
                        or native['length'] - native['target_start'] != int(idx['response_EOS_targets'])
                        or hashlib.sha256(raw.encode()).digest() != idx['source_sha256'].tobytes()):
                    raise ValueError('Every original complete word source and native EOS budget must match')
                word_bytes.append(len(row['response'].encode('utf8')))
        if len(word_bytes) != 91585 or sum(word_bytes) != 644063: raise ValueError('All original whole-word UTF8 targets are required')
        corrections_path = ROOT / 'data/public-document-visible-corrections-admitted-v1/all4-complete-corrections-and-preferences.jsonl'
        corrections = [json.loads(raw) for raw in corrections_path.open()]
        visual_rows = []; occurrences = Counter(); negative_total = 0
        for row in visual['entries']:
            name = row['family']; positions = row['original_local_positions']; native = family_rows[name][positions]
            if (int(native['context_tokens'].sum()) != row['complete_context_tokens']
                    or int(native['response_EOS_targets'].sum()) != row['response_EOS_targets']):
                raise ValueError('Every original complete visual batch must reconstruct from its exact index')
            negatives = [int(n) for n in native['correction_index'] if int(n) >= 0]
            negative_targets = 0
            for number in negatives:
                negative = corrections[number]['complete_original_negative_native']
                negative_targets += sum(t != -100 for t in negative['labels'][1:])
            negative_total += negative_targets; occurrences.update((name, p) for p in positions)
            visual_rows.append(dict(source_step=row['step'], epoch=0, family=name, original_positions=positions,
                complete_context_tokens=row['complete_context_tokens'], response_EOS_targets=row['response_EOS_targets'],
                rank_source_counts=row['proposed_rank_source_counts'], whole_word_UTF8_bytes=sum(word_bytes[p] for p in positions) if name == 'words' else 0,
                complete_original_negative_response_EOS_targets=negative_targets, complete_original_correction_indices=negatives,
                original_corrected_entry_sha256=digest(row)))
        from slm_lab.full_visual_source_admission_v1 import FAMILIES
        if occurrences != Counter({(name, p): 1 for name, count in FAMILIES.items() for p in range(count)}) or negative_total != 80:
            raise ValueError('Every original visual occurrence and all80negative targets must remain')
        streams['visual'] = dict(source_role='train', source_identity_sha256=sha(visual_path), seed=visual['seed'],
            whole_word_UTF8_bytes=644063, retained_complete_negative_targets=80, entries=visual_rows)
        write(OUT / 'visual.json', streams['visual'])
        write(OUT / 'progress.json', dict(owner=owner, phase='checking every original human-ranked pair', visual_views=302627))
        human_root = ROOT / 'data/whole-human-alignment-v2'; data = WholeHumanData(human_root, 'preferences', 'train', 'optimization')
        if len(data) != 85786: raise ValueError('All original human-ranked training pairs are required')
        order = EpochOrder(len(data), 959117); human_rows = []; chosen_targets = rejected_targets = chosen_context = rejected_context = 0
        for start in range(0, len(data), 16):
            positions = [order.at(p) for p in range(start, min(start + 16, len(data)))]
            info = [pair_information(data, p) for p in positions]
            chosen = sum(r['chosen_response_EOS_targets'] for r in info); rejected = sum(r['rejected_response_EOS_targets'] for r in info)
            cc = sum(data.rows[p]['native']['chosen']['length'] for p in positions)
            rc = sum(data.rows[p]['native']['rejected']['length'] for p in positions)
            chosen_targets += chosen; rejected_targets += rejected; chosen_context += cc; rejected_context += rc
            human_rows.append(dict(source_step=len(human_rows) + 1, epoch=0, family='human_preferences', original_positions=positions,
                rank_source_counts=[len(positions[r::4]) for r in range(4)], complete_context_tokens=cc + rc,
                response_EOS_targets=chosen + rejected, chosen_context_tokens=cc, rejected_context_tokens=rc,
                chosen_response_EOS_targets=chosen, rejected_response_EOS_targets=rejected,
                complete_original_pair_information_sha256=digest(info)))
            if len(human_rows) % 500 == 0:
                write(OUT / 'progress.json', dict(owner=owner, phase='checking every original human-ranked pair', human_batches=len(human_rows)))
        if Counter(p for row in human_rows for p in row['original_positions']) != Counter(range(85786)):
            raise ValueError('No original human pair can be dropped, duplicated or rounded')
        streams['human'] = dict(source_role='train', source_identity_sha256=digest(dict(manifest_sha256=sha(human_root/'manifest.json'),
            admission_sha256=sha(human_root/'admission.json'), seed=959117, original_rows_sha256=digest(human_rows))),
            seed=959117, human_ranked_other_model_responses_not_fresh_own_output_judgments=True,
            chosen_response_EOS_targets=chosen_targets, rejected_response_EOS_targets=rejected_targets,
            chosen_policy_context_tokens=chosen_context, DPO_policy_context_tokens=chosen_context + rejected_context,
            DPO_reference_context_tokens=chosen_context + rejected_context, entries=human_rows)
        write(OUT / 'human.json', streams['human'])
        for path in [visual_path, folder/'complete.json', folder/'source-training-admission.json', index_path, word_path, corrections_path,
                human_root/'manifest.json', human_root/'admission.json', human_root/'independent-source-readback.json',
                human_root/'preferences.train.jsonl', human_root/'preferences.train.tokens.u16',
                ROOT/'reports/mixed-coding-SFT-source-bindings-CPU-v1/complete.json']:
            files[str(path)] = sha(path)
        for path, expected in data.manifest['files_sha256'].items():
            if path.startswith('preferences.train.'):
                if sha(human_root/path) != expected: raise ValueError('The original complete human native pack changed')
        bound = FileBindings(files); bound.check(full=True)
        joint = combine(streams); require_original_scope(joint)
        joint['component_files'] = {name: dict(path=str(OUT/(name+'.json')), sha256=sha(OUT/(name+'.json'))) for name in streams}
        write(OUT / 'joint.json', joint)
        if torch.cuda.is_initialized(): raise ValueError('Source-only preparation cannot acquire a CUDA context')
        bound.check(full=True)
        write(OUT / 'complete.json', dict(complete=True, at=datetime.now(timezone.utc).isoformat(), owner=owner,
            source_files_sha256=files, joint_sha256=sha(OUT/'joint.json'), actual_joint_updates=5362,
            original_stream_budgets=joint['original_stream_budgets'], actual_task_combinations=joint['task_combinations'],
            full_schedule_data_only=True, independent_joint_readback_still_required=True, joint_training_admitted=False,
            models_constructed=0, neural_forwards=0, optimizer_updates=0, GPU_contexts=0, goal_achieved=False))
    except BaseException as exc:
        write(OUT / 'failure.json', dict(owner=owner, error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False, goal_achieved=False))
        raise


if __name__ == '__main__': main()
