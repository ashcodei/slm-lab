"""Index complete original later instructions; never launch neural training."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_later_instruction_replay_v2 import FAMILIES, KINDS, LaterInstructionReplay, source_family
from slm_lab.codenet_replay_v1 import INDEX_DTYPE, FullReplayEncoder
from slm_lab.complete_js_parameter_admission import admit as admit_js
from slm_lab.goruby_header_admission import admit as admit_goruby
from slm_lab.exercism_support_admission import admit as admit_exercises
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.model import ModelConfig


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'data/codenet-later-instruction-replay-v2'
REPORTS = ROOT / 'reports/codenet-later-instruction-replay-v2'
RESERVE = 16 * 1024**3
MAXIMUM = 64 * 1024**2
REPORT_CAP = 16 * 1024**2


def status(message, **fields):
    atomic_json(REPORTS / 'status.json', dict(status=message,
        updated_at=datetime.now(timezone.utc).isoformat(), goal_achieved=False, **fields))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only original source indexing required')
    if OUTPUT.exists() or REPORTS.exists():
        raise ValueError('Preserve earlier source preparation; no implicit restart')
    if shutil.disk_usage(ROOT).free < RESERVE + MAXIMUM + REPORT_CAP:
        raise ValueError('Fund complete64MiB indices and16MiB reports above16GiB persistent reserve')
    REPORTS.mkdir()
    status('reconstructing every original later exercise and complete source-header variant')
    try:
        token = ROOT / 'data/codenet-durable-pack-v2/tokenizer.json'
        config = ROOT / 'configs/model.json'
        directories = [ROOT / 'data' / name for name in ['exercism-support-context-v3',
            'complete-js-parameters-v1', 'complete-goruby-headers-v1']]
        paths = [Path(__file__), token, config] + [p / name for p in directories for name in ['manifest.json', 'train.jsonl']]
        paths += [ROOT / 'tests/test_codenet_later_instruction_replay_v2.py',
            ROOT / 'reports/codenet-later-instruction-replay-v2-focused-tests.log',
            ROOT / 'src/slm_lab/javascript_recovered_toolchain_v2.py',
            ROOT / 'reports/javascript-recovered-toolchain-v2-focused-tests.log',
            ROOT / 'reports/javascript-recovered-toolchain-v2-compatibility.json']
        paths += [ROOT / 'src/slm_lab' / (name + '.py') for name in ['codenet_later_instruction_replay_v2',
            'codenet_replay_v1', 'codenet_checkpoint_v1', 'complete_js_parameter_admission',
            'goruby_header_admission', 'exercism_support_admission', 'goruby_code_headers',
            'instruction_generalization_trial', 'model', 'training', 'corpus', 'diverse_images']]
        bound = {str(p): file_hash(p) for p in paths}
        bound.update({str(ROOT / p): digest for p, digest in capture()['files_sha256'].items()})
        guard = FileBindings(bound)
        receipts = {}
        for kind, admit in zip(KINDS, [admit_exercises, admit_js, admit_goruby]):
            rows, receipt = admit(ROOT)
            assert receipt['passed'] is True
            receipts[kind] = receipt
            status('unchanged original source admission completed', source_kind=kind, source_rows=len(rows))
            del rows
        guard.check(full=True)
        OUTPUT.mkdir()
        sources = []
        for kind, folder in zip(KINDS, directories):
            m = json.loads((folder / 'manifest.json').read_text())
            assert m['tokenizer_sha256'] == bound[str(token)]
            sources.append(dict(kind=kind, path=str(folder / 'train.jsonl'), sha256=bound[str(folder / 'train.jsonl')],
                manifest=str(folder / 'manifest.json'), manifest_sha256=bound[str(folder / 'manifest.json')], rows=m['train_examples']))
        encoder = FullReplayEncoder(Tokenizer.from_file(str(token)), ModelConfig.load(config), ROOT / 'data', 448)
        entries = []
        counts, contexts, targets = Counter(), Counter(), Counter()
        seen = set()
        with (OUTPUT / 'records.jsonl').open('x') as records:
            for source in sources:
                kind = source['kind']
                with Path(source['path']).open('rb') as stream:
                    while raw := stream.readline():
                        offset = stream.tell() - len(raw)
                        row = json.loads(raw)
                        name = source_family(row, kind)
                        if row['id'] in seen:
                            raise ValueError('Original source record IDs must remain unique')
                        seen.add(row['id'])
                        ids, labels = encoder.sequence(row, 'response')
                        count = sum(v != -100 for v in labels)
                        if 'actual_target_tokens' in row and row['actual_target_tokens'] != count:
                            raise ValueError('Original complete source target budget differs')
                        full_hash = hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).digest()
                        raw_hash = hashlib.sha256(raw).digest()
                        entries.append((offset, len(raw), FAMILIES.index(name), len(ids), count, np.void(raw_hash), np.void(full_hash)))
                        counts[name] += 1
                        contexts[name] += len(ids)
                        targets[name] += count
                        records.write(json.dumps(dict(id=row['id'], family=name, source_kind=kind,
                            source_offset=offset, source_bytes=len(raw), source_row_sha256=raw_hash.hex(),
                            full_input_ids_sha256=full_hash.hex(), context_tokens=len(ids), target_tokens=count)) + '\n')
                        if len(entries) % 1000 == 0:
                            records.flush()
                            if records.tell() + len(entries) * INDEX_DTYPE.itemsize > MAXIMUM or shutil.disk_usage(ROOT).free < RESERVE:
                                raise ValueError('Preserve bounded index commitment and persistent reserve')
                            guard.check()
                            status('indexing complete original source inputs without copying code', unique_source_rows=len(entries), kind=kind)
        assert len(entries) == 48584 and encoder.truncated == 0
        assert counts['complete_JS_parameter_headers'] == 982
        assert counts['complete_Go_source_headers'] == 11435 and counts['complete_Ruby_source_headers'] == 35571
        assert sum(counts[name] for name in FAMILIES[:6]) == 596
        assert sum(targets[name] for name in FAMILIES[:6]) == receipts['exercises']['complete_response_targets_per_pass']
        assert targets[FAMILIES[6]] == receipts['javascript']['actual_targets_per_pass']
        assert targets[FAMILIES[7]] + targets[FAMILIES[8]] == receipts['goruby']['actual_targets_per_source_pass']
        np.save(OUTPUT / 'index.npy', np.asarray(entries, dtype=INDEX_DTYPE), allow_pickle=False)
        admission = dict(passed=True, held_roles_excluded=True,
            source_manifests_sha256={s['kind']: s['manifest_sha256'] for s in sources}, receipts=receipts,
            publisher_reconstruction_executed_for_every_source_variant=True,
            source_program_execution_not_newly_claimed=True, goal_achieved=False)
        atomic_json(OUTPUT / 'source-admission.json', admission)
        artifacts = ['index.npy', 'records.jsonl', 'source-admission.json']
        size = sum((OUTPUT / name).stat().st_size for name in artifacts)
        assert size <= MAXIMUM and shutil.disk_usage(ROOT).free >= RESERVE
        repetitions = {name: 4 if i < 6 else 1 for i, name in enumerate(FAMILIES)}
        manifest = dict(format='slm-complete-later-instruction-replay-v2', completed_at=datetime.now(timezone.utc).isoformat(),
            sources=sources, tokenizer_sha256=bound[str(token)], config_sha256=bound[str(config)], image_size=448,
            family_names=list(FAMILIES), unique_family_counts=dict(counts),
            family_counts={name: counts[name] * repetitions[name] for name in FAMILIES},
            family_reference_repetitions=repetitions, unique_source_rows=len(entries), complete_fit_rows=50372,
            unique_source_context_tokens=dict(contexts), unique_source_response_targets=dict(targets),
            planned_response_target_exposures={name: targets[name] * repetitions[name] for name in FAMILIES},
            source_rows_copied=0, encoder_truncated_sequences=0,
            artifact_files_sha256={name: file_hash(OUTPUT / name) for name in artifacts}, stored_artifact_bytes=size,
            original_source_admission_rechecked=True, independent_new_index_readback_pending=True,
            final_base_match=False, all_later_sources_covered=False, production_runner_integration=False,
            new_neural_training_started=False, goal_achieved=False,
            scope='48584 original source rows and50372 explicit replay records, including four queued exposures '
                'of each of596 original exercises. Header variants reuse earlier original algorithms; '
                'repetitions and interface variants are not new unique programs. No source code copied/repaired, '
                'held data admitted, source semantic correctness inferred or model capability claimed.')
        guard.check(full=True)
        atomic_json(OUTPUT / 'manifest.json', manifest)
        reader = LaterInstructionReplay(OUTPUT)
        assert sum(reader.size(name) for name in reader.families) == 50372
        atomic_json(REPORTS / 'results.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=bound, manifest_sha256=file_hash(OUTPUT / 'manifest.json'),
            unique_source_rows=48584, explicit_replay_records=50372, family_counts=manifest['family_counts'],
            unique_source_response_targets=dict(targets), stored_artifact_bytes=size,
            original_admissions_reconstructed_every_source=True, independent_index_audit_pending=True,
            new_neural_training_started=False, goal_achieved=False))
        status('all complete later exercise/header sources indexed; independent index/schedule audit pending',
            unique_source_rows=48584, explicit_replay_records=50372, stored_artifact_bytes=size)
        print(json.dumps(dict(unique_source_rows=48584, replay_records=50372, stored_artifact_bytes=size)), flush=True)
    except Exception as error:
        status('source preparation stopped; preserve partial evidence and inspect', error=str(error))
        raise


if __name__ == '__main__':
    main()
