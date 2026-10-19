"""Rebuild every prepared native target from original publisher code and text.

Waits for its exact CPU preparation owner. Preserves candidate/held roles and
does not turn source labels into execution rewards or training admission.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from tokenizers import Tokenizer

from slm_lab.code_contests_curriculum_candidates_v1 import LANGUAGES, original_problem_keys, prompt, question_hash
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/code-contests-complete-curriculum-candidate-v1'
PRODUCER = ROOT / 'reports/code-contests-complete-curriculum-candidate-v1'
OUT = ROOT / 'reports/code-contests-complete-curriculum-v1-readback'


def alive(owner):
    path = Path('/proc') / str(owner['pid']) / 'stat'
    if not path.exists(): return False
    values = path.read_text().split(') ', 1)[1].split()
    return values[19] == owner['ticks'] and values[0] not in {'Z', 'X'}


def next_row(stream):
    line = stream.readline()
    return json.loads(line) if line else None


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('Fresh CPU-hidden whole-pack readback required')
    producer_plan_path = PRODUCER / 'plan.json'; producer_plan = json.loads(producer_plan_path.read_text())
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    files = {str(p): file_hash(p) for p in [producer_plan_path, *python_sources(ROOT, [Path(__file__)])]}
    queue_guard = FileBindings(files)
    OUT.mkdir()
    atomic_json(OUT / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=owner,
        original_preparation_owner=producer_plan['owner'], source_files_sha256=files,
        every_original_problem_and_every_selected_presentation_required=True,
        actual_BPE_context_and_EOS_target_masks_rebuilt=True,
        publisher_correctness_not_fresh_local_execution=True, dataset_admitted_to_training=False, model_updates=0))
    try:
        deadline = time.monotonic() + 24*3600
        while True:
            queue_guard.check(full=False)
            if (PRODUCER / 'failure.json').exists(): raise ValueError('Original preparation failed; preserve all original evidence')
            running = alive(producer_plan['owner'])
            if (PRODUCER / 'complete.json').exists() and not running: break
            if not running: raise ValueError('Original preparation owner terminated before completion')
            if time.monotonic() > deadline: raise ValueError('Reader observation deadline; no producer restart authorized or attempted')
            time.sleep(20)
        completion_path = PRODUCER / 'complete.json'; completion = json.loads(completion_path.read_text())
        if completion['dataset_admitted_to_training'] is not False: raise ValueError('Candidate-only original pack required')
        source_manifest_path = ROOT / 'data/code-contests-native-train-records-v1/manifest.json'
        source_manifest = json.loads(source_manifest_path.read_text())
        schema_path = ROOT / 'data/code-contests-protobuf-schema-v1/manifest.json'; schema = json.loads(schema_path.read_text())
        bound = {**files, **completion['source_files_sha256'], **source_manifest['files_sha256'],
            str(source_manifest_path): file_hash(source_manifest_path), str(schema_path): file_hash(schema_path)}
        guard = FileBindings(bound)
        pack = CompleteCodePack(DATA, 'train', 'audit')
        tokenizer = Tokenizer.from_file(str(DATA / 'tokenizer.json'))
        sys.path.insert(0, schema['isolated_reader_site_packages']); sys.path.insert(0, str(schema_path.parent))
        import contest_problem_pb2 as proto
        profiles = json.loads((PRODUCER / 'all-original-problem-source-dispositions.json').read_text())
        source_rows = [json.loads(line) for line in (ROOT / 'data/code-contests-native-train-records-v1/all-original-problem-records.jsonl').open()]
        if len(profiles) != len(source_rows) or len({r['id'] for r in profiles}) != len(source_rows):
            raise ValueError('Every original source problem and distinct identity required')
        context_counts = defaultdict(Counter); language_counts = defaultdict(Counter); source_roles = Counter()
        staged_count = 0; quarantined_count = 0; packed_count = 0; original_problem_count = 0
        source_stream = None; source_path = None
        per_language = Counter(); seen_ids = set()
        with gzip.open(PRODUCER / 'all-selected-original-presentation-candidates.jsonl.gz', 'rt') as stage_stream, \
                gzip.open(DATA / 'original-rows.jsonl.gz', 'rt') as packed_stream:
            stage = next_row(stage_stream)
            for source, profile in zip(source_rows, profiles):
                if source['id'] != profile['id']: raise ValueError('Original source order changed')
                if source['extracted_record_file'] != source_path:
                    if source_stream: source_stream.close()
                    source_path = source['extracted_record_file']; source_stream = open(source_path, 'rb')
                source_stream.seek(source['extracted_record_offset']); raw = source_stream.read(source['extracted_record_bytes'])
                if hashlib.sha256(raw).hexdigest() != source['actual_original_record_sha256']:
                    raise ValueError('Original complete publisher payload differs')
                problem = proto.ContestProblem()
                if problem.ParseFromString(raw) != len(raw): raise ValueError('Complete original protobuf parse required')
                if (profile['complete_original_question_normalized_sha256'] != question_hash(problem.description)
                        or profile['known_platform_identity_keys'] != [list(k) for k in original_problem_keys(source)]
                        or profile['original_name'] != problem.name):
                    raise ValueError('Original question and platform identity profile differs')
                protected = profile['component_protected_source_matches']; role = profile['split']
                expected_role = 'quarantine' if protected else ('train' if profile['known_prior_training_source_in_component'] else
                    ('validation' if int(hashlib.sha256(('own-CodeContests-role-v1:'+profile['group_id']).encode()).hexdigest()[:8],16)%20 == 0 else 'train'))
                if role != expected_role: raise ValueError('Fixed source-derived component role differs')
                source_roles[role] += 1; per_language.clear(); selected_here = 0
                while stage is not None and stage['original_problem_id'] == source['id']:
                    index = stage['original_solution_position']
                    if not 0 <= index < len(problem.solutions): raise ValueError('Selected position is not an original positive')
                    original = problem.solutions[index]
                    language = int(original.language)
                    if (language not in LANGUAGES or stage['language'] != LANGUAGES[language][1]
                            or stage['original_language_number'] != language or stage['response'] != original.solution
                            or stage['prompt'] != prompt(problem, language)
                            or stage['original_program_sha256'] != hashlib.sha256(original.solution.encode()).hexdigest()):
                        raise ValueError('Selected complete original positive, language or full question differs')
                    prefix = [1] + tokenizer.encode('User: ' + stage['prompt'] + '\nAssistant:').ids
                    response = tokenizer.encode(original.solution).ids
                    ids = prefix + response + [2]; labels = [-100]*len(prefix) + response + [2]
                    if (len(ids) > 4096 or not response or any(v < 4 for v in prefix[1:] + response)
                            or stage['ids'] != ids or stage['labels'] != labels):
                        raise ValueError('Fresh complete native BPE IDs and whole-response/EOS mask differ')
                    identity = source['id'] + ':solution:' + str(index)
                    if identity in seen_ids: raise ValueError('Repeated selected source position')
                    seen_ids.add(identity); per_language[language] += 1; selected_here += 1; staged_count += 1
                    if role == 'quarantine': quarantined_count += 1
                    else:
                        row = next_row(packed_stream)
                        if row is None or row['id'] != identity or row['split'] != role or row['publisher_split'] != 'train':
                            raise ValueError('Every retained candidate and sealed original role required')
                        for key in ['original_solution_position','language','original_language_number','original_program_sha256','prompt','response']:
                            if row[key] != stage[key]: raise ValueError('Durable original source target text differs')
                        entry = pack.index[packed_count]
                        offset = int(entry['offset']); actual = pack.ids[offset:offset+int(entry['length'])]
                        start = int(entry['target_start'])
                        if (int(entry['role']) != {'train':0,'validation':1}[role] or start != len(prefix)
                                or not np.array_equal(actual, np.asarray(ids,dtype='<u2')) or row['token_pack_index'] != packed_count
                                or row['complete_input_ids_sha256'] != hashlib.sha256(actual.tobytes()).hexdigest()
                                or row['target_tokens'] != len(response)+1):
                            raise ValueError('Every durable native token, role, target boundary and actual digest must reconstruct')
                        for key, value in [('examples',1),('context_tokens',len(ids)),('target_tokens',len(response)+1)]: context_counts[role][key] += value
                        language_counts[role][row['language']] += 1; packed_count += 1
                    stage = next_row(stage_stream)
                if selected_here != profile['candidate_presentations'] or any(v > 3 for v in per_language.values()):
                    raise ValueError('Every selected original presentation and fixed language cap required')
                original_problem_count += 1
                if original_problem_count % 1024 == 0:
                    atomic_json(OUT / 'progress.json', dict(original_problems_checked=original_problem_count,
                        complete_packed_targets_rebuilt=packed_count, quarantined_presentations_checked=quarantined_count, model_updates=0))
                    print(json.dumps(dict(problems=original_problem_count, packed_targets=packed_count)), flush=True)
            if stage is not None or next_row(packed_stream) is not None: raise ValueError('Unaccounted staged or packed native source records')
        if source_stream: source_stream.close()
        expected_counts = {role:dict(context_counts[role]) if context_counts[role] else dict(examples=0,context_tokens=0,target_tokens=0)
                           for role in ['train','validation','test']}
        if (packed_count != len(pack.index) or packed_count != pack.manifest['source_examples']
                or expected_counts != pack.manifest['counts'] or dict(source_roles) != completion['actual_original_problem_dispositions']
                or quarantined_count != completion['quarantined_selected_presentations']
                or staged_count != completion['actual_original_solution_dispositions']['selected_complete_original_programs']):
            raise ValueError('Whole original corpus, native pack and quarantine accounting differs')
        guard.check(full=True); queue_guard.check(full=True)
        atomic_json(OUT / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), complete=True,
            original_problems_checked=original_problem_count, actual_staged_presentations_rebuilt=staged_count,
            quarantined_original_presentations_rebuilt=quarantined_count,
            retained_native_presentations_rebuilt=packed_count, actual_native_pack_counts=expected_counts,
            actual_native_pack_languages={k:dict(v) for k,v in language_counts.items()},
            all_selected_actual_publisher_code_and_complete_question_bytes_and_BPE_tokens_and_EOS_masks_rebuilt=True,
            source_exposure_negative_results_and_selection_search_not_independently_rescanned=True,
            original_publisher_correctness_not_new_local_execution=True,
            further_source_overlap_and_execution_reward_admission_pending=True,
            source_files_sha256={str(p):file_hash(p) for p in [OUT / 'plan.json', completion_path, DATA / 'manifest.json']},
            dataset_admitted_to_training=False, model_updates=0, candidate_promoted=False, goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT / 'complete.json'), actual_native_targets=packed_count)), flush=True)
    except BaseException as error:
        atomic_json(OUT / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
            dataset_admitted_to_training=False, model_updates=0, goal_achieved=False))
        raise


if __name__ == '__main__': main()
