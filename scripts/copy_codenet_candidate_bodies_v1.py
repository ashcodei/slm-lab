"""Copy the exact original candidate sources and duplicate metadata into owned cache."""
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
import sys

from slm_lab.codenet_candidate_bodies_v1 import copy_bodies
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'reports/codenet-candidate-bodies-v1'
    if folder.exists():
        raise ValueError('Preserve earlier original source copying')
    folder.mkdir()
    cache = None
    def status(message, **fields):
        atomic_json(folder / 'status.json', {'status': message,
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'source_code_executed': False, 'dataset_admitted_to_training': False,
            'goal_achieved': False, **fields})
    try:
        source_path = ROOT / 'data/codenet-original-source-v1/source.json'
        inspection_path = ROOT / 'reports/codenet-original-inspection-v1/inspection.json'
        manifest_path = ROOT / 'data/codenet-candidate-selection-v1/manifest.json'
        source = json.loads(source_path.read_text())
        inspection = json.loads(inspection_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        selected_path = manifest_path.with_name('selected-metadata.jsonl')
        if not inspection['whole_gzip_stream_drained_crc_checked'] or manifest['inspection_sha256'] != file_hash(inspection_path):
            raise ValueError('Exact completed source inspection required')
        if source['sha256'] != inspection['archive_sha256'] or file_hash(selected_path) != manifest['selected_metadata_sha256']:
            raise ValueError('Original source or selected metadata differs')
        archive = Path(source['archive'])
        if archive.parent.resolve() != Path(inspection['metadata_cache']).parent.resolve() or archive.name != 'Project_CodeNet.tar.gz':
            raise ValueError('Original owned source archive location differs')
        cache = archive.parent / 'candidate-bodies-v1'
        if shutil.disk_usage(archive.parent).free < 13 * 1024 ** 3:
            raise ValueError('Require bounded source store plus temporary model reserve')
        selected = {}
        with selected_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                m = row['metadata']
                expected = str(PurePosixPath('Project_CodeNet', 'data', m['problem_id'], m['language'], m['submission_id'] + '.' + m['filename_ext']))
                if row['archive_member'] != expected or expected in selected or m['status'] != 'Accepted' or row['split'] is not None:
                    raise ValueError('Exact accepted source member/identity differs')
                selected[expected] = int(m['code_size'])
        if len(selected) != manifest['counts']['selected_candidates']:
            raise ValueError('Incomplete original selected candidate population')
        other_index = Path(inspection['metadata_cache']) / 'other-members.jsonl'
        if file_hash(other_index) != inspection['other_member_index_sha256']:
            raise ValueError('Original auxiliary member inventory changed')
        prefix = 'Project_CodeNet/derived/duplicates/'
        required = {prefix + 'README', prefix + 'identical_problem_clusters'}
        for language in ['C', 'C++', 'Java', 'Python']:
            base = prefix + language + '/Project_CodeNet-' + language
            required.update(base + '.' + suffix for suffix in ['clusters', 'graph', 'stats', 'sh'])
            required.update([base + '-tokenize.conf', prefix + language + '/' + language + '_accepted_outliers.txt',
                             prefix + language + '/duplicates.log.csv'])
        auxiliary = {}
        for line in other_index.open():
            row = json.loads(line)
            if row['member'] in required:
                if row['member'] in auxiliary:
                    raise ValueError('Ambiguous original auxiliary inventory')
                auxiliary[row['member']] = row['bytes']
        if set(auxiliary) != required:
            raise ValueError('Original problem/duplicate metadata members missing')
        paths = [Path(__file__), ROOT / 'src/slm_lab/codenet_candidate_bodies_v1.py',
                 ROOT / 'src/slm_lab/codenet_archive_inspection_v1.py', source_path,
                 inspection_path, manifest_path, selected_path, other_index]
        bound = {path: file_hash(path) for path in paths}
        bound[archive] = source['sha256']
        states = {path: (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns) for path in bound}
        def guard(full=False):
            for path, digest in bound.items():
                current = path.stat()
                if (current.st_ino, current.st_size, current.st_mtime_ns) != states[path]:
                    raise ValueError('Bound source copying input changed')
                if full and file_hash(path) != digest:
                    raise ValueError('Exact source copying input bytes differ')
        status('verifying original source bytes before selected-body copy')
        guard(full=True)
        atomic_json(folder / 'plan.json', {'recorded_at': datetime.now(timezone.utc).isoformat(),
            'bound_files_sha256': {str(p): d for p, d in bound.items()}, 'python': sys.version,
            'original_archive': str(archive), 'candidate_cache': str(cache),
            'selected_source_bodies': len(selected), 'auxiliary_members': auxiliary,
            'maximum_retained_source_store_GiB': 1, 'maximum_body_MiB': 1,
            'minimum_free_temporary_GiB': 12, 'maximum_decompressed_GiB': 128,
            'source_code_executed': False, 'split_assignment_completed': False,
            'model_training_started': False, 'goal_achieved': False})
        def progress(fields):
            guard()
            status('copying selected original bodies and duplicate metadata; no execution', **fields)
            print(json.dumps(fields), flush=True)
        result = copy_bodies(archive, cache, selected, auxiliary, progress)
        guard(full=True)
        if result['members'] != inspection['members'] or result['original_code_headers'] != inspection['section_regular_file_counts']['data'] or result['decompressed_bytes'] != inspection['decompressed_bytes']:
            raise ValueError('Complete original archive population/stream differs')
        if file_hash(cache / 'source-bodies.bin') != result['source_body_store_sha256']:
            raise ValueError('Independent source body store digest differs')
        for record in result['auxiliary_members'].values():
            if file_hash(cache / record['cache_relative_path']) != record['sha256']:
                raise ValueError('Independent auxiliary source digest differs')
        atomic_json(folder / 'bodies.json', {'completed_at': datetime.now(timezone.utc).isoformat(),
            **result, 'candidate_cache': str(cache), 'source_archive_sha256': source['sha256'],
            'selected_metadata_sha256': manifest['selected_metadata_sha256'],
            'source_index_sha256': file_hash(cache / 'source-index.jsonl'),
            'goal_achieved': False,
            'scope': 'Exact selected original bytes and original duplicate metadata only. Judge labels do not establish current-runtime correctness, efficiency, clean splits or model capability. Context/target eligibility and family/clone handling remain pending.'})
        status('selected original bodies and duplicate metadata copied; dataset admission pending')
    except Exception as error:
        status('selected original source copying stopped; requires inspection', error=str(error), candidate_cache=str(cache))
        raise


if __name__ == '__main__':
    main()
