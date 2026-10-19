"""Read-only original-source, image and whole-query audit; never starts CUDA."""
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import time

import google_crc32c
from PIL import Image, ImageOps
from tokenizers import Tokenizer

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.localized_pointer_evidence_v1 import expected_rows, original_queries, require_row, dense_target
from slm_lab.vqa_pairs import known_photo_aliases

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/localized-pointer-candidate-v1'
SOURCE = ROOT/'data/localized-pointer-source-v1'
REPORTS = ROOT/'reports/localized-pointer-candidate-v1'
TOKEN = ROOT/'checkpoints/dialogue-retention-v1/systems-variety-code/tokenizer.json'
PARENTS = [('train', 'expanded-everyday-photos-v1/train.jsonl'),
    ('train', 'everyday-images-v1/train.jsonl'), ('validation', 'everyday-images-v1/valid.jsonl')]


def identity(pid):
    path = Path('/proc')/str(pid)/'stat'
    if not path.exists():
        return None
    fields = path.read_text().split(') ', 1)[1].split()
    return dict(pid=pid, ticks=fields[19], state=fields[0])


def live(pid, ticks):
    actual = identity(pid)
    return actual is not None and actual['ticks'] == ticks and actual['state'] not in ('Z', 'X')


def records(path):
    with Path(path).open() as source:
        for line in source:
            yield json.loads(line)


def parents_and_metadata(plan):
    parents = {'train': {}, 'validation': {}}
    for role, name in PARENTS:
        path = ROOT/'data'/name
        for row in records(path):
            key = row['source_image_id']
            if row.get('task') != 'caption' or row.get('source') != 'Open Images / CVDF' or key in parents[role]:
                continue
            if row['source_subset'] != role:
                raise ValueError('Original parent file roles must match original publisher roles')
            image = (path.parent/row['image']).resolve()
            if not image.is_relative_to(ROOT/'data'):
                raise ValueError('Only retained original project images allowed')
            parents[role][key] = {**row, '_path': str(image), '_parent': str(path)}
    if parents['train'].keys() & parents['validation'].keys():
        raise ValueError('Original photo identities cross source roles')
    metadata = {'train': {}, 'validation': {}}
    for role in metadata:
        with (ROOT/'data/image-source-cache-v1'/('openimages-'+role+'.csv')).open(newline='') as source:
            for row in csv.DictReader(source):
                if row['ImageID'] in parents[role]:
                    metadata[role][row['ImageID']] = row
    blocked = set()
    for role in ('train', 'valid'):
        for pair in records(ROOT/'data/vqa-source-cache-v1'/('candidate-pairs.'+role+'.jsonl')):
            for side in ('first', 'second'):
                blocked.update(known_photo_aliases(pair[side]['image']))
    held_aliases = set().union(*(known_photo_aliases(p) for p in parents['validation'].values()))
    return parents, metadata, blocked, held_aliases


def pixels(path):
    """Recompute both actual encoder inputs and dHash without producer helpers."""
    with Image.open(path) as image:
        dimensions = list(image.size)
        hashes = {n: hashlib.sha256(ImageOps.pad(image.convert('RGB'), (n, n),
            method=Image.Resampling.BICUBIC, color=(128,128,128)).tobytes()).hexdigest() for n in (224,448)}
        gray = list(image.convert('L').resize((9,8), Image.Resampling.BILINEAR).getdata())
        fingerprint = 0
        for y in range(8):
            for x in range(8):
                fingerprint = (fingerprint << 1) | int(gray[y*9+x+1] > gray[y*9+x])
    return dimensions, hashes, fingerprint


def check_photo(photo, parent, metadata, blocked, held_aliases, protected):
    role, key = photo['split'], photo['image_id']
    aliases = known_photo_aliases(parent)
    if aliases & blocked or (role == 'train' and aliases & held_aliases):
        raise ValueError('Protected photo aliases entered a candidate')
    if (metadata['License'].rstrip('/') != 'https://creativecommons.org/licenses/by/2.0'
            or not metadata.get('Rotation') or float(metadata['Rotation']) != 0 or not metadata['Author']):
        raise ValueError('Original explicit rotation and creator/license metadata required')
    mappings = dict(author='Author', author_profile_url='AuthorProfileURL', source_url='OriginalLandingURL', source_title='Title')
    if any(photo[k] != parent[k] or photo[k] != metadata[v] for k,v in mappings.items()):
        raise ValueError('Original author, title and attribution URLs must remain exact')
    if (photo['image_license'] != 'CC-BY-2.0' or photo['image_license_url'] != metadata['License']
            or photo['image_license_url'] != parent['license_url'] or photo['image'] != parent['_path']
            or photo['original_parent_records'] != parent['_parent'] or photo['group_id'] != parent['group_id']
            or photo['source_photo_aliases'] != sorted(aliases)
            or photo['original_EXIF_orientation'] != 1 or photo['publisher_rotation_degrees'] != 0):
        raise ValueError('Original photo, credit, identity and role evidence must be preserved')
    if file_hash(photo['image']) != parent['stored_image_sha256'] or photo['stored_image_sha256'] != parent['stored_image_sha256']:
        raise ValueError('Actual complete stored original image changed')
    dimensions, hashes, fingerprint = pixels(photo['image'])
    if (dimensions != photo['stored_dimensions'] or hashes[224] != photo['model_pixel_sha256']
            or hashes[448] != photo['model_pixel_sha256_448'] or hashes[224] != parent['model_pixel_sha256']
            or (parent.get('model_pixel_sha256_448') and hashes[448] != parent['model_pixel_sha256_448'])
            or str(fingerprint) != photo['difference_hash64']):
        raise ValueError('Every actual model pixel/hash must reconstruct')
    if role == 'train' and (hashes[224] in protected[224] or hashes[448] in protected[448]
            or any((fingerprint ^ other).bit_count() <= 4 for other in protected['fingerprints'])):
        raise ValueError('Protected validation pixels or perceptual neighbors entered training')
    header = DATA/photo['source_header']
    if header.resolve() != (DATA/'headers'/(key+'.jpg.header')).resolve():
        raise ValueError('Original per-photo retained header path required')
    raw = header.read_bytes()
    if not 0 < len(raw) <= 131072 or hashlib.sha256(raw).hexdigest() != photo['source_header_sha256']:
        raise ValueError('Actual bounded original JPEG header bytes required')
    with Image.open(io.BytesIO(raw)) as image:
        original = list(image.size)
        if (image.format != 'JPEG' or image.getexif().get(274, 1) != 1
                or min(original) < 96 or original[0]*original[1] > 10_000_000):
            raise ValueError('Only original upright bounded JPEG scenes supply geometry')
    if (original != photo['original_CVDF_dimensions']
            or abs(dimensions[0]*original[1]-dimensions[1]*original[0]) > max(original)):
        raise ValueError('Stored scene must preserve the original source aspect')
    provenance = photo['source_header_provenance']
    if provenance['mode'] == 'exact_original_cached_download':
        cache = Path(provenance['cache'])
        expected = ROOT/'data/image-source-cache-v1/source-images'/role/(key+'.jpg')
        if cache != expected or file_hash(cache) != parent['download_sha256'] or provenance['original_download_sha256'] != parent['download_sha256']:
            raise ValueError('Actual original cached full download identity required')
        with cache.open('rb') as source:
            if source.read(131072) != raw:
                raise ValueError('Retained header differs from actual full original JPEG')
    elif provenance['mode'] == 'original_publisher_range_header':
        match = re.fullmatch(r'bytes 0-(\d+)/(\d+)', provenance.get('content_range') or '')
        if (provenance['url'] != parent['download_url'] or provenance['etag'] != parent['download_etag']
                or provenance['complete_original_download_bytes_rehashed'] is not False
                or (match and (int(match[1])+1 != len(raw) or int(match[2]) < len(raw)))
                or (not match and provenance.get('content_range') is not None)):
            raise ValueError('Retained partial-header receipt must bind original publisher object')
    else:
        raise ValueError('Unknown source header provenance')


def audit():
    manifest = json.loads((DATA/'manifest.json').read_text())
    plan = json.loads((DATA/'preparation-plan.json').read_text())
    collection = json.loads((SOURCE/'collection.json').read_text())
    bindings = {**plan['parent_files_sha256'], str(SOURCE/'collection.json'):plan['source_collection_sha256'],
        str(ROOT/'scripts/prepare_localized_pointer_candidates_v1.py'):plan['source_implementation_sha256'],
        str(ROOT/'src/slm_lab/localized_pointer_v1.py'):plan['span_implementation_sha256'],
        str(TOKEN):manifest['tokenizer_sha256'],
        **{str(DATA/name):sha for name,sha in manifest['files_sha256'].items()}}
    for name, expected in bindings.items():
        if file_hash(name) != expected:
            raise ValueError('Original candidate/source bytes changed: '+name)
    for name, receipt in collection['sources'].items():
        sha = hashlib.sha256(); crc = google_crc32c.Checksum(); length = 0
        with (SOURCE/name).open('rb') as source:
            while block := source.read(1 << 20):
                sha.update(block); crc.update(block); length += len(block)
        if sha.hexdigest() != receipt['sha256'] or length != receipt['bytes']:
            raise ValueError('Complete original publisher source length/hash changed')
        if receipt['publisher_crc32c'] and base64.b64encode(crc.digest()).decode() != receipt['publisher_crc32c']:
            raise ValueError('Original publisher checksum differs from actual full source')
    parents, metadata, blocked, held_aliases = parents_and_metadata(plan)
    protected = {224:set(), 448:set(), 'fingerprints':[]}
    for parent in parents['validation'].values():
        if file_hash(parent['_path']) != parent['stored_image_sha256']:
            raise ValueError('Original protected validation source changed')
        _, hashes, fingerprint = pixels(parent['_path'])
        for n in (224,448):
            protected[n].add(hashes[n])
        protected['fingerprints'].append(fingerprint)
    photos = {}
    for photo in records(DATA/'photos.jsonl'):
        key = photo['split'], photo['image_id']
        if key in photos or key[0] not in parents or key[1] not in parents[key[0]]:
            raise ValueError('Every eligible photo must have one original source identity')
        photos[key] = photo
        check_photo(photo, parents[key[0]][key[1]], metadata[key[0]][key[1]], blocked, held_aliases, protected)
        if len(photos) % 200 == 0:
            print('independent_original_photo_pixels_headers_credits_checked', len(photos), flush=True)
    qualifications = list(records(DATA/'photo-qualifications.jsonl'))
    excluded = {(p['split'],p['image_id']) for p in qualifications}
    if len(excluded) != len(qualifications) or excluded & photos.keys() or any(not p.get('qualified_out') or not p.get('reason') for p in qualifications):
        raise ValueError('Every source photo has one eligible or preserved qualified-out outcome')
    actual = {}
    for role in parents:
        for row in records(DATA/(role+'.jsonl')):
            if row['id'] in actual or row['split'] != role or (role,row['image_id']) not in photos:
                raise ValueError('Every original query must be unique and in its original file role')
            dense_target(row, photos[role,row['image_id']]['stored_dimensions'])
            actual[row['id']] = row
    tokenizer = Tokenizer.from_file(str(TOKEN))
    if [tokenizer.token_to_id(t) for t in ('<pad>','<bos>','<eos>')] != [0,1,2]:
        raise ValueError('Own native query framing vocabulary required')
    planned = set(); seen = set(); joins = Counter(); rows = Counter(); points = Counter(); reviews = Counter()
    for role,name in [('train','train-shard00000.jsonl'),('validation','validation.jsonl')]:
        path = SOURCE/name
        with path.open('rb') as source:
            while True:
                offset = source.tell(); raw = source.readline()
                if not raw:
                    break
                joins[role+':original_annotation_rows'] += 1
                if len(raw) > 2*1024**2:
                    joins[role+':oversized_original_record_review'] += 1
                    continue
                record = json.loads(raw); key = record['image_id']; parent = parents[role].get(key)
                if parent is None:
                    continue
                aliases = known_photo_aliases(parent)
                if aliases & blocked or (role == 'train' and aliases & held_aliases):
                    joins[role+':protected_source_photo_alias'] += 1
                    continue
                if key not in metadata[role]:
                    joins[role+':missing_original_metadata'] += 1
                    continue
                try:
                    queries = original_queries(record)
                except (ValueError,KeyError,TypeError):
                    joins[role+':original_transcript_or_trace_review'] += 1
                    continue
                if not queries:
                    joins[role+':no_complete_stable_content_span'] += 1
                    continue
                joins[role+':matched_original_annotation_rows'] += 1
                planned.add((role,key))
                if (role,key) not in photos:
                    continue
                expected, reviewed = expected_rows(record, raw, path, offset, photos[role,key], tokenizer)
                if reviewed:
                    reviews[role+':whole_query_budget_or_reserved_marker'] += reviewed
                for row in expected:
                    if row['id'] in seen or row['id'] not in actual:
                        raise ValueError('Every complete original query must occur exactly once')
                    seen.add(row['id']); require_row(actual[row['id']], row)
                    rows[role] += 1; points[role] += row['source_point_count']
    if (seen != actual.keys() or planned != photos.keys() | excluded
            or dict(joins) != plan['original_annotation_join_counts'] or dict(joins) != manifest['original_annotation_join_counts']
            or dict(rows) != manifest['complete_query_rows'] or dict(points) != manifest['original_attention_point_samples']
            or dict(reviews) != manifest['whole_query_reviews'] or len(planned) != plan['unique_matched_original_images']
            or len(planned) != manifest['photo_headers_checked'] or len(photos) != manifest['eligible_original_photos']
            or dict(Counter(role for role,_ in photos)) != manifest['eligible_photos_by_split']):
        raise ValueError('Every original source, qualified photo, whole query and point population must reconcile')
    for name, expected in bindings.items():
        if file_hash(name) != expected:
            raise ValueError('Bound original source changed during independent audit')
    paths = [Path(__file__).resolve(), ROOT/'src/slm_lab/localized_pointer_evidence_v1.py',
        ROOT/'tests/test_localized_pointer_evidence_v1.py', ROOT/'reports/localized-pointer-evidence-v1-tests.log',
        SOURCE/'alignment-qualification.json', DATA/'manifest.json']
    result = dict(completed_at=datetime.now(timezone.utc).isoformat(), complete_query_rows=dict(rows),
        original_attention_point_samples=dict(points), original_annotation_join_counts=dict(joins),
        original_photo_candidates=len(planned), actual_eligible_photos=len(photos), qualified_out_photos=len(excluded),
        every_whole_original_query_source_record_offset_sha_times_and_point_index_reconstructed=True,
        every_query_BPE_BOS_content_EOS_and_448px_14x14_grid_count_reconstructed=True,
        every_actual_photo_header_EXIF_dimensions_pixels_creator_and_role_rechecked=True,
        all_original_validation_800_photos_protected_by_alias_exact224_448px_and_dHash_distance4=True,
        all_original_VQA_candidate_aliases_excluded=True,
        complete_original_sources_SHA_length_and_publisher_CRC32C_verified=True,
        excluded_network_errors_recorded_without_reissuing_requests=True,
        range_header_receipts_bind_original_ETag_but_do_not_rehash_complete_original_download=True,
        original_temporal_alignment_speech_model_assisted=True,
        approximate_annotator_attention_not_object_box_or_OCR_truth=True,
        original_development_validation_not_independent_final_test=True,
        source_candidate_files_sha256=bindings, audit_files_sha256={str(p):file_hash(p) for p in paths},
        candidate_manifest_sha256=file_hash(DATA/'manifest.json'),
        training_admitted=False, model_updates=0, capability_improvement_established=False, goal_achieved=False)
    atomic_json(REPORTS/'independent-source-readback.json', result)
    print('independent_complete_original_pointer_source_readback', json.dumps(dict(rows)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-pid', type=int)
    parser.add_argument('--wait-for-ticks')
    args = parser.parse_args()
    if bool(args.wait_for_pid) != bool(args.wait_for_ticks):
        raise ValueError('Exact original producer identity required')
    if (REPORTS/'independent-source-readback.json').exists() or (REPORTS/'audit-queue.json').exists():
        raise ValueError('Preserve previous independent audit/queue evidence')
    if '2 passed' not in (ROOT/'reports/localized-pointer-evidence-v1-tests.log').read_text():
        raise ValueError('Actual independent source/tamper/geometry tests required')
    REPORTS.mkdir(exist_ok=True)
    import os
    paths = [Path(__file__).resolve(), ROOT/'src/slm_lab/localized_pointer_evidence_v1.py',
        ROOT/'tests/test_localized_pointer_evidence_v1.py', ROOT/'reports/localized-pointer-evidence-v1-tests.log']
    bound = {str(p):file_hash(p) for p in paths}
    atomic_json(REPORTS/'audit-queue.json', dict(queued_at=datetime.now(timezone.utc).isoformat(),
        actor=identity(os.getpid()), exact_original_producer=dict(pid=args.wait_for_pid,ticks=args.wait_for_ticks),
        source_files_sha256=bound, read_only=True, no_CUDA_or_process_signals=True, model_updates=0, goal_achieved=False))
    while not (DATA/'manifest.json').exists():
        if not args.wait_for_pid or not live(args.wait_for_pid,args.wait_for_ticks):
            raise ValueError('Original producer ended without final candidate; preserve partial evidence')
        time.sleep(5)
    if any(file_hash(p) != sha for p,sha in bound.items()):
        raise ValueError('Original independent auditor changed while queued')
    audit()


if __name__ == '__main__':
    main()
