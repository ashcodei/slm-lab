"""Wait for the exact original download, then inspect metadata without executing code."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from slm_lab.codenet_archive_inspection_v1 import inspect_archive
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def process(pid):
    directory = Path(f'/proc/{pid}')
    try:
        raw = (directory / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        argv = [x.decode() for x in (directory / 'cmdline').read_bytes().split(bytes([0])) if x]
        return None if fields[0] == 'Z' else (fields[19], argv)
    except FileNotFoundError:
        return None


def main():
    folder = ROOT / 'reports/codenet-original-inspection-v1'
    if folder.exists():
        raise ValueError('Preserve earlier source inspection')
    folder.mkdir()
    source = ROOT / 'data/codenet-original-source-v1'
    launch_path = ROOT / 'reports/codenet-original-source-v1-launch.json'
    launch = json.loads(launch_path.read_text())
    bound = {path: file_hash(path) for path in [Path(__file__),
        ROOT / 'src/slm_lab/codenet_archive_inspection_v1.py', launch_path, source / 'plan.json']}
    def guard():
        if any(file_hash(path) != digest for path, digest in bound.items()):
            raise ValueError('Bound inspection source/plan changed')
    def status(message, **fields):
        atomic_json(folder / 'status.json', {'status': message,
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'dataset_admitted_to_training': False, 'goal_achieved': False, **fields})
    try:
        while not (source / 'source.json').exists():
            guard()
            if process(launch['pid']) != (launch['start_ticks'], launch['argv']):
                raise ValueError('Original acquisition ended without a completed source receipt')
            status('waiting for verified live original source acquisition', source_pid=launch['pid'],
                   source_start_ticks=launch['start_ticks'])
            time.sleep(15)
        receipt = json.loads((source / 'source.json').read_text())
        plan = json.loads((source / 'plan.json').read_text())
        archive = Path(receipt['archive'])
        if archive.resolve() != (Path(plan['archive_cache']) / 'Project_CodeNet.tar.gz').resolve():
            raise ValueError('Original owned archive location differs')
        if receipt['bytes'] != plan['publisher_expected_bytes'] or receipt['etag'] != plan['publisher_etag']:
            raise ValueError('Original publisher byte/object identity differs')
        for name, digest in receipt['supporting_file_sha256'].items():
            if file_hash(source / name) != digest:
                raise ValueError('Original source/license evidence changed')
        status('verifying complete original compressed source bytes')
        if archive.stat().st_size != receipt['bytes'] or file_hash(archive) != receipt['sha256']:
            raise ValueError('Complete original source bytes differ')
        bound[source / 'source.json'] = file_hash(source / 'source.json')
        identity = archive.stat()
        cache = archive.parent / 'inspection-v1'
        if shutil.disk_usage(archive.parent).free < 15 * 1024 ** 3:
            raise ValueError('Require bounded metadata plus temporary model reserve')
        atomic_json(folder / 'plan.json', {'recorded_at': datetime.now(timezone.utc).isoformat(),
            'archive': str(archive), 'archive_sha256': receipt['sha256'],
            'source_receipt_sha256': bound[source / 'source.json'],
            'bound_files_sha256': {str(p.relative_to(ROOT)): d for p, d in bound.items()},
            'metadata_cache': str(cache), 'maximum_retained_metadata_GiB': 3,
            'maximum_decompressed_GiB': 128, 'minimum_free_temporary_GiB': 12,
            'source_submission_execution': False, 'dataset_admitted_to_training': False,
            'goal_achieved': False})
        def progress(fields):
            guard()
            actual = archive.stat()
            if (actual.st_ino, actual.st_size, actual.st_mtime_ns) != (identity.st_ino, identity.st_size, identity.st_mtime_ns):
                raise ValueError('Original archive changed during inspection')
            status('inspecting original archive; code remains unexecuted', **fields)
            print(json.dumps(fields), flush=True)
        result = inspect_archive(archive, cache, progress)
        guard()
        if file_hash(archive) != receipt['sha256']:
            raise ValueError('Final original source digest differs')
        atomic_json(folder / 'inspection.json', {
            'completed_at': datetime.now(timezone.utc).isoformat(), **result,
            'archive_sha256': receipt['sha256'], 'metadata_cache': str(cache),
            'retained_member_index_sha256': file_hash(cache / 'retained-members.jsonl'),
            'other_member_index_sha256': file_hash(cache / 'other-members.jsonl'),
            'goal_achieved': False,
            'scope': 'Complete archive/metadata inspection only; no program correctness, efficiency, curated split or model capability claim.'})
        status('original archive and metadata inspection completed; dataset preparation pending')
    except Exception as error:
        status('original source inspection stopped; requires inspection', error=str(error))
        raise


if __name__ == '__main__':
    main()
