"""Read every original training prompt before future full shared-native PPO.

Pure source work: no torch import, neural forward, GPU query, lease or update.
The complete full human source is retained; held pairs never select a capacity
case. This recipe does not stand in for a trained head or actual GPU capacity.
"""
import argparse
import ast
from array import array
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import mmap
import os
from pathlib import Path
import sys

from slm_lab.local_source_closure_v2 import python_sources

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/whole-human-alignment-v2'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text())


def options():
    tree = ast.parse((ROOT/'src/slm_lab/native_shared_ppo_v1.py').read_text())
    node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'OPTIONS' for t in n.targets))
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'dict' or node.args:
        raise ValueError('Exact literal fixed full PPO settings are required without importing neural code')
    return {k.arg: ast.literal_eval(k.value) for k in node.keywords}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path,
        default=Path('/tmp/slm-native-shared-PPO-source-recipe-CPU-v1'))
    out = parser.parse_args().output.absolute()
    if (out.exists() or out.resolve() != out or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or sys.byteorder != 'little' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])
            or 'torch' in sys.modules):
        raise ValueError('A fresh one-thread CUDA-hidden idle source-only recipe is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    manifest, admission = read(DATA/'manifest.json'), read(DATA/'admission.json')
    files = [DATA/name for name in ['manifest.json', 'admission.json', 'independent-source-readback.json',
        'preferences.train.jsonl', 'preferences.train.tokens.u16', 'preferences.valid.jsonl', 'preferences.valid.tokens.u16']]
    files += python_sources(ROOT, [Path(__file__), ROOT/'src/slm_lab/native_shared_ppo_v1.py'])
    files = sorted(set(files))
    states = {str(p): (p.stat().st_dev, p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in files}
    hashes = {str(p): sha(p) for p in files}
    if (manifest.get('format') != 'whole-original-human-conversation-and-ranking-packed-v2'
            or manifest.get('full_original_responses_and_EOS_preserved') is not True
            or admission.get('training_admitted') is not True
            or admission.get('original_held_gradients_forbidden') is not True
            or admission.get('source_manifest_sha256') != hashes[str(DATA/'manifest.json')]
            or admission.get('independent_readback_sha256') != hashes[str(DATA/'independent-source-readback.json')]
            or manifest['splits']['preferences.train']['examples'] != 85786
            or manifest['splits']['preferences.valid']['examples'] != 4642
            or any(hashes[str(DATA/name)] != manifest['files_sha256'][name] for name in
                ['preferences.train.jsonl', 'preferences.train.tokens.u16', 'preferences.valid.jsonl', 'preferences.valid.tokens.u16'])):
        raise ValueError('The whole original train/held source and independently admitted unmodified token packs are required')
    settings = options()
    if (settings['steps'] != 42893 or settings['prompts_per_update'] != 2 or settings['max_new_tokens'] != 512):
        raise ValueError('Retain the complete original85786prompt/512action epoch')
    counts = Counter(); maximum = None; position = 0; cursor = 0; context = 0; prompts = 0
    identity = hashlib.sha256(); seen = set()
    with (DATA/'preferences.train.tokens.u16').open('rb') as raw, (DATA/'preferences.train.jsonl').open() as index:
        with mmap.mmap(raw.fileno(), 0, access=mmap.ACCESS_READ) as tokens:
            if len(tokens) % 2: raise ValueError('Every original little-endian native token byte must be complete')
            for line in index:
                row = json.loads(line)
                if (row['split'] != 'train' or row['original_split'] != 'train' or row['feedback_source'] != 'human'
                        or row.get('whole_original_response_whitespace_preserved') is not True
                        or set(row['native']) != {'chosen', 'rejected'} or row['id'] in seen):
                    raise ValueError('Every distinct original human train pair must retain its exact source role and whole responses')
                seen.add(row['id']); sequences = {}; boundaries = []
                for field in ['chosen', 'rejected']:
                    info = row['native'][field]
                    if (set(info) != {'offset', 'length', 'target_start'} or info['offset'] != cursor
                            or any(type(v) is not int for v in info.values())
                            or not 1 <= info['target_start'] < info['length'] <= manifest['config']['max_text_length']):
                        raise ValueError('Every whole original response offset, prompt boundary and length must remain unchanged')
                    values = array('H', tokens[2*cursor:2*(cursor+info['length'])])
                    if (len(values) != info['length'] or values[0] != 1 or values[-1] != 2
                            or max(values) >= manifest['config']['vocab_size']):
                        raise ValueError('Every complete original native BOS/response/EOS and vocabulary value must remain')
                    sequences[field] = values; boundaries.append(info['target_start'])
                    cursor += len(values); context += len(values)
                boundary = boundaries[0]
                if boundaries[1] != boundary or sequences['chosen'][:boundary] != sequences['rejected'][:boundary]:
                    raise ValueError('Every chosen/rejected source must have the same unchanged whole prompt')
                prompts += boundary; counts[row['source']] += 1
                item = dict(original_train_position=position, id=row['id'], source=row['source'],
                    source_record_sha256=row['source_record_sha256'], complete_prompt_tokens=boundary,
                    prompt_tokens_u16_sha256=hashlib.sha256(sequences['chosen'][:boundary].tobytes()).hexdigest())
                observation = {**item, **{f+'_complete_native_u16_sha256': hashlib.sha256(v.tobytes()).hexdigest()
                    for f, v in sequences.items()}}
                identity.update((json.dumps(observation, sort_keys=True)+'\n').encode())
                if maximum is None or boundary > maximum['complete_prompt_tokens']: maximum = item
                position += 1
                if position % 8192 == 0: print(json.dumps(dict(read_complete_original_train_pairs=position)), flush=True)
            if position != 85786 or 2*cursor != len(tokens):
                raise ValueError('Every complete original human source pair and native token must be covered exactly once')
    if (maximum['complete_prompt_tokens']+settings['max_new_tokens']+1 > manifest['config']['max_text_length']
            or settings['steps']*settings['prompts_per_update'] != position):
        raise ValueError('The full whole-prompt/action/scoring horizon must fit without truncating or replaying any source')
    for p, state in states.items():
        s = Path(p).stat()
        if (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) != state:
            raise ValueError('Every bound original source byte must remain unchanged through the full read')
    if 'torch' in sys.modules: raise ValueError('No neural import is permitted in this source recipe')
    complete = dict(complete=True, at=datetime.now(timezone.utc).isoformat(),
        format='own-native-shared-PPO-complete-original-source-recipe-CPU-v1', source_files_sha256=hashes,
        full_original_train_pairs=position, original_held_pairs_kept_separate=4642,
        every_complete_original_prompt_and_both_responses_read=True, source_counts=dict(counts),
        complete_original_chosen_and_rejected_context_tokens=context, whole_original_prompt_tokens=prompts,
        all_original_sources_and_whole_prompt_identity_sha256=identity.hexdigest(), longest_original_train_prompt=maximum,
        maximum_prompt_plus512actions_and_scoring_EOS=maximum['complete_prompt_tokens']+513,
        options=settings, unrepeated_source_epoch=True, repeated_rounding_prompts=0,
        held_pairs_used_for_capacity_or_gradients=0, requires_actual_learned_current_native_reward_bundle=True,
        requires_actual_full_size_GPU_capacity_and_recovery=True, shared_PPO_production_updates=0,
        neural_forwards=0, GPU_queries=0, GPU_contexts=0, production_GPU_admitted=False,
        useful_own_output_alignment_proved=False, goal_achieved=False)
    out.mkdir()
    with (out/'complete.json').open('x') as stream:
        stream.write(json.dumps(complete, indent=2)+'\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({k: complete[k] for k in ['complete', 'full_original_train_pairs',
        'maximum_prompt_plus512actions_and_scoring_EOS', 'shared_PPO_production_updates']}), flush=True)


if __name__ == '__main__': main()
