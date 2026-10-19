"""Extract every verified official training record with the original proto schema.

Preserves original protobuf bytes, including tests and both correct/incorrect
human solutions. Correctness here is publisher metadata, not fresh execution.
No generated model sample, external pretrained weight or held split is used.
This creates a lossless source index, not an optimization-ready corpus.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.verified_simple_riegeli_v2 import VerifiedSimpleRiegeli

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/code-contests-original-train-cache-v2'
SCHEMA = ROOT / 'data/code-contests-protobuf-schema-v1'
OUT = ROOT / 'data/code-contests-native-train-records-v1'
REPORTS = ROOT / 'reports/code-contests-native-train-records-v1'


def sha(data): return hashlib.sha256(data).hexdigest()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists() or REPORTS.exists(): raise ValueError('Fresh CPU-hidden complete original-record extraction required')
    if shutil.disk_usage(ROOT).free < 60 * 1024**3: raise ValueError('Preserve original writer reserve and whole uncompressed corpus budget')
    source_path = SOURCE / 'manifest.json'; source = json.loads(source_path.read_text())
    schema_path = SCHEMA / 'manifest.json'; schema = json.loads(schema_path.read_text())
    if (source['original_split'] != 'train' or source['original_train_shards'] != 128 or source['training_admitted'] is not False
            or schema['original_publisher_schema_sha256'] != source['source_metadata']['contest_problem.proto']['sha256']):
        raise ValueError('Complete original publisher train cache and exact official schema required')
    files = {str(p): file_hash(p) for p in [Path(__file__), source_path, schema_path,
        ROOT / 'src/slm_lab/verified_simple_riegeli_v2.py', ROOT / 'tests/test_verified_simple_riegeli_v2.py', ROOT / 'reports/verified-simple-riegeli-v2-tests.log',
        ROOT / '.tools/code-contests-protobuf-reader-v1/requirements.lock', ROOT / 'reports/code-contests-protobuf-reader-v1-pip-resolution.json']}
    files.update({str(SCHEMA / name): value for name, value in schema['files_sha256'].items()})
    files.update({r['path']: r['actual_SHA256'] for r in source['complete_original_shards']}); guard = FileBindings(files)
    # Isolate these imports to this CPU data-reader process. No production
    # training environment files or already running processes are modified.
    sys.path.insert(0, schema['isolated_reader_site_packages']); sys.path.insert(0, str(SCHEMA))
    import contest_problem_pb2 as proto
    import google.protobuf
    if sha(proto.DESCRIPTOR.serialized_pb) == '': raise ValueError('Original compiled descriptor required')
    OUT.mkdir(); (OUT / 'records').mkdir(); REPORTS.mkdir(); started = time.monotonic()
    owner = dict(pid=os.getpid(), ticks=(Path('/proc') / str(os.getpid()) / 'stat').read_text().split(') ', 1)[1].split()[19])
    atomic_json(REPORTS / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=owner,
        source_files_sha256=files, original_train_shards=128, original_compressed_bytes=source['original_train_bytes'],
        exact_original_schema_descriptor_sha256=sha(proto.DESCRIPTOR.serialized_pb), official_protobuf_version=google.protobuf.__version__,
        original_proto_record_bytes_preserved=True, accepted_and_incorrect_solutions_kept_separate=True,
        maximum_whole_decoded_chunk_bytes=512 * 1024**2,
        original_whole_file_SHA256_integrity_checked_internal_HighwayHash_not_computed=True,
        unchanged_signature_block_and_chunk_extents_and_all_size_vectors_verified=True,
        metadata_chunks_encode_zero_problem_records_and_are_not_training_examples=True,
        training_admitted=False, model_updates=0, goal_achieved=False))
    counts = Counter(); languages = dict(correct=Counter(), incorrect=Counter()); files_out = {}; readers = []
    names = Counter(); identities = set(); index = OUT / 'all-original-problem-records.jsonl'
    try:
        with index.open('x') as metadata:
            for shard_index, original in enumerate(source['complete_original_shards']):
                reader = VerifiedSimpleRiegeli(original['path'], original['actual_SHA256'])
                destination = OUT / 'records' / f'{shard_index:05d}.original-records.pb'; offset = 0; local_records = 0
                with destination.open('xb') as stream:
                    for original_bytes, location in reader:
                        problem = proto.ContestProblem(); consumed = problem.ParseFromString(original_bytes)
                        if consumed != len(original_bytes): raise ValueError('Every complete original protobuf record must be parsed')
                        # Original proto2 presence and all unknown fields are
                        # retained by the standard implementation. Keep raw
                        # bytes regardless of any serialization field order.
                        roundtrip = problem.SerializeToString(deterministic=True)
                        identical = roundtrip == original_bytes
                        counts['byte_identical_standard_protobuf_roundtrip_records'] += int(identical)
                        counts['standard_protobuf_roundtrip_size_changed_records'] += int(len(roundtrip) != len(original_bytes))
                        if not isinstance(problem.name, str) or not isinstance(problem.description, str):
                            raise ValueError('Original problem identity/description needs explicit byte-encoding review')
                        counts['problem_records'] += 1; counts['original_proto_bytes'] += len(original_bytes)
                        counts['description_UTF8_bytes'] += len(problem.description.encode())
                        tests = {kind: len(getattr(problem, kind)) for kind in ['public_tests', 'private_tests', 'generated_tests']}
                        for kind, total in tests.items(): counts[kind] += total
                        correct = len(problem.solutions); incorrect = len(problem.incorrect_solutions)
                        counts['publisher_correct_solutions'] += correct; counts['publisher_incorrect_solutions'] += incorrect
                        counts['records_without_correct_solutions'] += int(correct == 0)
                        counts['records_without_description'] += int(not problem.description)
                        per_language = dict(correct=Counter(), incorrect=Counter())
                        for kind, values in [('correct', problem.solutions), ('incorrect', problem.incorrect_solutions)]:
                            for solution in values:
                                language = proto.ContestProblem.Solution.Language.Name(solution.language)
                                per_language[kind][language] += 1; languages[kind][language] += 1
                                code = solution.solution
                                if isinstance(code, str): counts[kind + '_solution_UTF8_bytes'] += len(code.encode())
                                elif isinstance(code, bytes): counts[kind + '_solution_non_UTF8_fields'] += 1; counts[kind + '_solution_raw_non_UTF8_bytes'] += len(code)
                                else: raise ValueError('Original source text byte representation requires explicit review')
                        original_sha = sha(original_bytes); identity = 'CodeContests:publisher-train:' + original_sha
                        if identity in identities: raise ValueError('Repeated complete original record requires explicit duplicate disposition')
                        identities.add(identity); names[(int(problem.source), problem.name)] += 1
                        stream.write(original_bytes)
                        row = dict(id=identity, original_split='train', original_source=proto.ContestProblem.Source.Name(problem.source),
                            original_name=problem.name, original_file=original['name'], original_file_SHA256=original['actual_SHA256'],
                            original_record_location=location, actual_original_record_sha256=original_sha,
                            extracted_record_file=str(destination), extracted_record_offset=offset, extracted_record_bytes=len(original_bytes),
                            publisher_correct_solutions=correct, publisher_incorrect_solutions=incorrect,
                            original_language_counts={k: dict(v) for k, v in per_language.items()}, original_test_counts=tests,
                            description_UTF8_bytes=len(problem.description.encode()),
                            standard_protobuf_byte_identical_roundtrip=identical,
                            original_CF_contest_id=problem.cf_contest_id if problem.HasField('cf_contest_id') else None,
                            original_CF_problem_index=problem.cf_index if problem.HasField('cf_index') else None,
                            original_input_file=problem.input_file if problem.HasField('input_file') else None,
                            original_output_file=problem.output_file if problem.HasField('output_file') else None,
                            publisher_solution_labels_not_new_local_execution=True, training_admitted=False)
                        metadata.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
                        offset += len(original_bytes); local_records += 1
                    stream.flush(); os.fsync(stream.fileno())
                if local_records != reader.records or offset != reader.decoded_bytes: raise ValueError('Every original shard record and decompressed protobuf byte must be retained')
                files_out[str(destination)] = file_hash(destination)
                readers.append(dict(original_file=original['name'], actual_original_records=reader.records,
                    original_chunks=reader.chunks, chunk_types=reader.types, actual_decoded_bytes=reader.decoded_bytes,
                    zstd_library=str(reader.zstd.path), zstd_library_sha256=reader.zstd.library_sha256))
                metadata.flush(); os.fsync(metadata.fileno())
                progress = dict(owner=owner, original_shards_completed=len(readers), original_total_shards=128,
                    actual_counts=dict(counts), actual_language_counts={k: dict(v) for k, v in languages.items()},
                    elapsed_seconds=time.monotonic() - started, training_admitted=False, model_updates=0, goal_achieved=False)
                atomic_json(REPORTS / 'progress.json', progress)
                if len(readers) % 8 == 0: print(json.dumps(progress), flush=True)
        if len(readers) != 128 or len(identities) != counts['problem_records']: raise ValueError('Every original128-shard problem record must be accounted for')
        guard.check(full=True); files_out[str(index)] = file_hash(index)
        repeats = [{'original_source': source_, 'original_name': name, 'records': count} for (source_, name), count in names.items() if count > 1]
        atomic_json(REPORTS / 'all-original-shard-readbacks.json', readers)
        atomic_json(REPORTS / 'same-platform-name-repetitions.json', repeats)
        atomic_json(OUT / 'manifest.json', dict(format='own-all-original-CodeContests-train-protobuf-records-v1',
            at=datetime.now(timezone.utc).isoformat(), original_train_shards=128, actual_counts=dict(counts),
            actual_original_solution_languages={k: dict(v) for k, v in languages.items()}, original_source_manifest_sha256=file_hash(source_path),
            original_source_cache=str(SOURCE), original_schema_manifest_sha256=file_hash(schema_path),
            original_schema_directory=str(SCHEMA), original_proto_record_bytes_and_all_test_fields_preserved=True,
            standard_protobuf_parser_used=True, internal_Riegeli_HighwayHash_not_computed_whole_publisher_file_SHA256_verified=True,
            same_platform_name_repetition_groups=len(repeats), original_record_ids_include_exact_payload_sha256=True,
            files_sha256=files_out, actual_token_packing_context_execution_and_source_overlap_admission_pending=True,
            publisher_correctness_labels_not_new_local_execution=True, Python2_and_Python3_labels_not_merged=True,
            no_validation_or_test_dataset_shards_used=True, training_admitted=False, model_updates=0, goal_achieved=False))
        atomic_json(REPORTS / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), complete=True,
            actual_original_train_shards=128, actual_original_problem_records=counts['problem_records'], actual_counts=dict(counts),
            actual_original_solution_languages={k: dict(v) for k, v in languages.items()},
            original_record_bytes_preserved_without_rewriting_or_discarding_fields=True,
            source_files_sha256={str(p): file_hash(p) for p in [REPORTS / 'plan.json', OUT / 'manifest.json', index,
                REPORTS / 'all-original-shard-readbacks.json', REPORTS / 'same-platform-name-repetitions.json']},
            actual_token_packing_context_execution_and_source_overlap_admission_pending=True,
            training_admitted=False, model_updates=0, candidate_promoted=False, goal_achieved=False))
        print(json.dumps(dict(complete=str(REPORTS / 'complete.json'), actual_original_problems=counts['problem_records'],
            correct_solutions=counts['publisher_correct_solutions'], incorrect_solutions=counts['publisher_incorrect_solutions'],
            actual_context_and_execution_and_overlap_admission_pending=True)), flush=True)
    except BaseException as error:
        atomic_json(REPORTS / 'failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(),
            training_admitted=False, model_updates=0, goal_achieved=False)); raise


if __name__ == '__main__': main()
