"""Read every extracted original record back against its publisher shard.

Re-derives the entire original problem/solution/test inventory using the official
protobuf implementation. Does not admit targets or execute/train a model.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.verified_simple_riegeli_v2 import VerifiedSimpleRiegeli

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/code-contests-native-train-records-v1'
OUT = ROOT / 'reports/code-contests-all-original-records-v1-readback'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('Fresh CPU-only complete source readback required')
    manifest_path = DATA / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source_path = Path(manifest['original_source_cache']) / 'manifest.json'
    source = json.loads(source_path.read_text())
    schema_path = Path(manifest['original_schema_directory']) / 'manifest.json'
    schema = json.loads(schema_path.read_text())
    completion_path = ROOT / 'reports/code-contests-native-train-records-v1/complete.json'
    completion = json.loads(completion_path.read_text())
    if not completion['complete'] or manifest['training_admitted'] is not False:
        raise ValueError('Complete lossless original source inventory required')
    files = {str(p): file_hash(p) for p in [Path(__file__), manifest_path, source_path, schema_path,
        completion_path, ROOT / 'src/slm_lab/verified_simple_riegeli_v2.py']}
    files.update(manifest['files_sha256'])
    files.update({r['path']: r['actual_SHA256'] for r in source['complete_original_shards']})
    files.update({str(schema_path.parent / n): h for n, h in schema['files_sha256'].items()})
    guard = FileBindings(files)
    sys.path.insert(0, schema['isolated_reader_site_packages'])
    sys.path.insert(0, str(schema_path.parent))
    import contest_problem_pb2 as proto
    OUT.mkdir()
    started = time.monotonic()
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    atomic_json(OUT / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=owner,
        source_files_sha256=files, all_original_shards_required=128,
        scope='Every raw publisher record versus every extracted raw record and official-schema metadata',
        same_structural_Riegeli_decoder_used=True, original_whole_file_SHA256_verified=True,
        internal_HighwayHash_not_computed=True, models_constructed=0, model_updates=0))
    counts = Counter(); languages = dict(correct=Counter(), incorrect=Counter())
    shards = []; identities = set(); names = Counter()
    index_path = DATA / 'all-original-problem-records.jsonl'
    try:
        with index_path.open() as index:
            for shard_index, original in enumerate(source['complete_original_shards']):
                path = DATA / 'records' / f'{shard_index:05d}.original-records.pb'
                digest = hashlib.sha256(); local = 0; offset = 0
                reader = VerifiedSimpleRiegeli(original['path'], original['actual_SHA256'])
                with path.open('rb') as extracted:
                    for raw, location in reader:
                        line = index.readline()
                        if not line: raise ValueError('Original problem index ended early')
                        row = json.loads(line); actual = extracted.read(len(raw))
                        if actual != raw: raise ValueError('Extracted record differs from its complete original publisher bytes')
                        raw_sha = hashlib.sha256(actual).hexdigest()
                        expected_id = 'CodeContests:publisher-train:' + raw_sha
                        if expected_id in identities: raise ValueError('Repeated whole-record identity requires explicit review')
                        identities.add(expected_id)
                        expected = dict(id=expected_id, original_split='train', original_file=original['name'],
                            original_file_SHA256=original['actual_SHA256'], original_record_location=location,
                            extracted_record_file=str(path), extracted_record_offset=offset,
                            extracted_record_bytes=len(actual), actual_original_record_sha256=raw_sha,
                            publisher_solution_labels_not_new_local_execution=True, training_admitted=False)
                        if any(row.get(k) != v for k, v in expected.items()):
                            raise ValueError('Every original identity, boundary and source location must match')
                        problem = proto.ContestProblem()
                        if problem.ParseFromString(actual) != len(actual): raise ValueError('Incomplete original protobuf parse')
                        identical = problem.SerializeToString(deterministic=True) == actual
                        if identical is not row['standard_protobuf_byte_identical_roundtrip']:
                            raise ValueError('Original protobuf roundtrip metadata differs')
                        counts['byte_identical_standard_protobuf_roundtrip_records'] += identical
                        counts['standard_protobuf_roundtrip_size_changed_records'] += int(len(problem.SerializeToString(deterministic=True)) != len(actual))
                        counts['problem_records'] += 1; counts['original_proto_bytes'] += len(actual)
                        counts['description_UTF8_bytes'] += len(problem.description.encode())
                        counts['records_without_correct_solutions'] += int(not problem.solutions)
                        counts['records_without_description'] += int(not problem.description)
                        names[(int(problem.source), problem.name)] += 1
                        if (row['original_name'] != problem.name or row['original_source'] != proto.ContestProblem.Source.Name(problem.source)
                                or row['description_UTF8_bytes'] != len(problem.description.encode())):
                            raise ValueError('Original problem identity/description metadata differs')
                        for key, field in [('original_CF_contest_id', 'cf_contest_id'), ('original_CF_problem_index', 'cf_index'),
                                ('original_input_file', 'input_file'), ('original_output_file', 'output_file')]:
                            if row[key] != (getattr(problem, field) if problem.HasField(field) else None):
                                raise ValueError('Original field presence or payload differs')
                        for kind in ['public_tests', 'private_tests', 'generated_tests']:
                            counts[kind] += len(getattr(problem, kind))
                            if row['original_test_counts'][kind] != len(getattr(problem, kind)):
                                raise ValueError('Original test inventory differs')
                        for kind, field in [('correct', 'solutions'), ('incorrect', 'incorrect_solutions')]:
                            values = getattr(problem, field); per_language = Counter()
                            counts['publisher_' + kind + '_solutions'] += len(values)
                            if row['publisher_' + kind + '_solutions'] != len(values): raise ValueError('Original solution count differs')
                            for solution in values:
                                language = proto.ContestProblem.Solution.Language.Name(solution.language)
                                languages[kind][language] += 1; per_language[language] += 1
                                code = solution.solution
                                if isinstance(code, str): counts[kind + '_solution_UTF8_bytes'] += len(code.encode())
                                elif isinstance(code, bytes):
                                    counts[kind + '_solution_non_UTF8_fields'] += 1
                                    counts[kind + '_solution_raw_non_UTF8_bytes'] += len(code)
                                else: raise ValueError('Original solution encoding requires review')
                            if dict(per_language) != row['original_language_counts'][kind]:
                                raise ValueError('Python2/Python3 or other original language counts differ')
                        digest.update(actual); offset += len(actual); local += 1
                    if extracted.read(1): raise ValueError('Unindexed extracted source bytes')
                if digest.hexdigest() != manifest['files_sha256'][str(path)]: raise ValueError('Entire extracted shard digest differs')
                shards.append(dict(shard=shard_index, records=local, original_bytes=offset,
                    extracted_and_publisher_bytes_identical=True))
                atomic_json(OUT / 'progress.json', dict(owner=owner, original_shards_checked=len(shards),
                    actual_counts=dict(counts), elapsed_seconds=time.monotonic() - started, model_updates=0))
                if len(shards) % 16 == 0:
                    print(json.dumps(dict(shards_checked=len(shards), records=counts['problem_records'])), flush=True)
            if index.readline(): raise ValueError('Extra extracted source index entries')
        actual_languages = {k: dict(v) for k, v in languages.items()}
        if (len(shards) != 128 or dict(counts) != manifest['actual_counts'] or actual_languages != manifest['actual_original_solution_languages']
                or sum(v > 1 for v in names.values()) != manifest['same_platform_name_repetition_groups']):
            raise ValueError('Whole original corpus inventory differs')
        guard.check(full=True)
        atomic_json(OUT / 'all128-shards-readback.json', shards)
        atomic_json(OUT / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), complete=True,
            every_original_record_and_extracted_byte_compared=True, complete_original_train_shards=128,
            actual_counts=dict(counts), actual_original_solution_languages=actual_languages,
            original_schema_parse_and_all_identity_boundary_test_language_counts_rederived=True,
            same_Riegeli_structural_decoder_not_independent_format_implementation=True,
            publisher_correctness_labels_not_new_local_execution=True, training_admitted=False,
            source_files_sha256={str(p): file_hash(p) for p in [OUT / 'plan.json', OUT / 'all128-shards-readback.json', manifest_path]},
            model_updates=0, candidate_promoted=False, goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT / 'complete.json'), actual_problem_records=counts['problem_records'])), flush=True)
    except BaseException as error:
        atomic_json(OUT / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), model_updates=0))
        raise


if __name__ == '__main__': main()
