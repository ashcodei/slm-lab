"""Acquire the original IBM archive for subsequent bounded coding preparation.

This downloads source data, never pretrained weights, and executes no submissions.
It does not admit code to the active training plan or label judge outcomes as RLHF.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

from slm_lab.codenet_source_download_v1 import download
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.public_sources import fetch

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz'
BYTES = 8343137473
ETAG = '"74257cf20935cb2827f01a9652eaa1b9-1592"'
REVISION = '55c323b527ab3d6510d55edff188bc0a0d7bc5e5'


def main():
    folder = ROOT / 'data/codenet-original-source-v1'
    if folder.exists():
        raise ValueError('Inspect earlier acquisition; no duplicate or automatic restart')
    if shutil.disk_usage('/tmp').free < BYTES + 12 * 1024 ** 3:
        raise ValueError('Require complete archive plus temporary model reserve')
    if shutil.disk_usage(ROOT).free < 26.25 * 1024 ** 3:
        raise ValueError('Preserve live persistent storage reserve')
    folder.mkdir()
    cache = Path(tempfile.mkdtemp(prefix='slm-codenet-original-source-v1-', dir='/tmp'))
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'source_url': URL,
        'publisher_expected_bytes': BYTES, 'publisher_etag': ETAG,
        'etag_is_not_a_whole_file_cryptographic_checksum': True,
        'archive_cache': str(cache), 'minimum_temporary_free_GiB': 12,
        'publisher_repository_revision': REVISION,
        'dataset_license': 'CDLA-Permissive-2.0',
        'dataset_license_evidence': 'Author paper Appendix A.2 and A.6; repository tooling is separately Apache-2.0.',
        'author_paper_url': 'https://jiechenjiechen.github.io/pub/codenet.pdf',
        'new_model_training': False, 'active_frozen_training_plan_changed': False,
        'submissions_executed': False, 'goal_achieved': False,
        'scope': 'Original source acquisition only. Subsequent admission requires accepted-status selection, source/description identity, whole problem-family splits, deduplication, complete contexts and constrained execution checks where supported.'}
    atomic_json(folder / 'plan.json', plan)
    def progress(offset, digest, retry):
        atomic_json(folder / 'progress.json', {
            'status': 'waiting before retrying original source' if retry else 'downloading original CodeNet source; unprepared and untrained',
            'updated_at': datetime.now(timezone.utc).isoformat(), 'received_bytes': offset,
            'expected_bytes': BYTES, 'prefix_sha256': digest, 'retry': retry,
            'cache': str(cache), 'goal_achieved': False})
    try:
        for name, url, limit in [
            ('publisher-README.md', f'https://raw.githubusercontent.com/IBM/Project_CodeNet/{REVISION}/README.md', 1000000),
            ('author-paper.pdf', plan['author_paper_url'], 20000000),
            ('DATASET-LICENSE.txt', 'https://raw.githubusercontent.com/spdx/license-list-data/main/text/CDLA-Permissive-2.0.txt', 1000000)]:
            (folder / name).write_bytes(fetch(url, limit=limit))
        if URL not in (folder / 'publisher-README.md').read_text():
            raise ValueError('Pinned publisher documentation does not identify this archive')
        license_text = (folder / 'DATASET-LICENSE.txt').read_text()
        if 'Community Data License Agreement' not in license_text or '2.0' not in license_text:
            raise ValueError('Dataset license text differs')
        progress(0, None, None)
        partial = cache / 'Project_CodeNet.tar.gz.partial'
        receipt = download(URL, partial, BYTES, ETAG, progress)
        if partial.stat().st_size != BYTES or file_hash(partial) != receipt['sha256']:
            raise ValueError('Completed local source bytes differ from streamed digest')
        archive = cache / 'Project_CodeNet.tar.gz'
        partial.rename(archive)
        atomic_json(folder / 'source.json', {**receipt, 'source_url': URL,
            'archive': str(archive), 'publisher_repository_revision': REVISION,
            'supporting_file_sha256': {name: file_hash(folder / name) for name in
                ['publisher-README.md', 'author-paper.pdf', 'DATASET-LICENSE.txt', 'plan.json']},
            'downloader_sha256': file_hash(ROOT / 'src/slm_lab/codenet_source_download_v1.py'),
            'script_sha256': file_hash(Path(__file__)),
            'completed_at': datetime.now(timezone.utc).isoformat(),
            'tar_gzip_integrity_and_dataset_admission_pending': True,
            'model_training_started': False, 'goal_achieved': False})
        atomic_json(folder / 'progress.json', {'status': 'original source downloaded; archive inspection and dataset preparation pending',
            'updated_at': datetime.now(timezone.utc).isoformat(), 'received_bytes': BYTES,
            'expected_bytes': BYTES, 'archive': str(archive), 'model_training_started': False, 'goal_achieved': False})
    except Exception as error:
        atomic_json(folder / 'progress.json', {'status': 'source acquisition stopped; inspect without automatic restart',
            'updated_at': datetime.now(timezone.utc).isoformat(), 'error': str(error),
            'cache': str(cache), 'goal_achieved': False})
        raise


if __name__ == '__main__':
    main()
