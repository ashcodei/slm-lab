"""Fresh full original-row/image/native-view readback of the document audit.

Independently scans actual requested pixels instead of reusing the producer's
extrema-based rule. It does not rerender every PDF or certify nonwhite labels.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource

from PIL import Image,ImageOps

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.local_source_closure_v2 import python_sources

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'reports/full-public-document-visible-labels-CPU-v1'
OUT = ROOT/'reports/full-public-document-visible-labels-v1-readback'
DATA = ROOT/'data/document-images-v1'
METADATA = ROOT/'reports/full-multimodal-source-epoch-CPU-v1/all302627-original-native-views.jsonl'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def atomic(path,value):
    pending = path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def state(path):
    s = Path(path).stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh complete original document/image audit readback on bounded low-priority CPU required')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0))
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    proof = read(SOURCE/'complete.json');vetoes = read(SOURCE/'semantic-vetoes.json')
    if (not proof.get('complete') or not proof['all_original_source_reconstructions_match'] or proof['training_admitted']
            or proof['actual_counts']['original_rows'] != 1433 or proof['actual_counts']['prepared_native_views_covered'] != 1382
            or vetoes['evidence_sha256'] != sha(SOURCE/'complete.json')):
        raise ValueError('The full original1433/1382 source audit and its exact unchanged veto evidence required')
    files = dict(proof['source_files_sha256'])
    paths = [SOURCE/'complete.json',SOURCE/'semantic-vetoes.json',SOURCE/'all1433-original-train-label-observations.jsonl',
        *python_sources(ROOT,[Path(__file__)])]
    files.update({str(p):sha(p) for p in paths})
    if files[str(SOURCE/'all1433-original-train-label-observations.jsonl')] != proof['all_original_dispositions_sha256']:
        raise ValueError('The entire original source observation stream must be complete and byte exact')
    if any(sha(p) != digest for p,digest in files.items()):raise ValueError('Every original runtime/source/image/PDF byte must match')
    states = {p:state(p) for p in files}
    def guard():
        if os.sched_getscheduler(0) != os.SCHED_IDLE or any(state(p) != s for p,s in states.items()):
            raise ValueError('Readback priority and every original source file must remain unchanged')
    def checked():
        wait_for_reserve(ROOT,minimum_bytes=40*1024**3,guard=guard)
        values = {k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
        used = values['anon']+values['shmem']+values['kernel'];limit = int(Path('/sys/fs/cgroup/memory.max').read_text())
        resident = next(int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
        if limit-used < 64*1024**3 or resident > 2*1024**3:raise ValueError('Bounded source-reader memory and other-process headroom required')
    checked();OUT.mkdir();owner = dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic(OUT/'plan.json',dict(owner=owner,source_files_sha256=files,original_rows=1433,prepared_native_views=1382,
        scheduler='SCHED_IDLE',every_requested_pixel_independently_scanned=True,every_PDF_not_rerendered_by_this_reader=True,
        nonwhite_transcript_quality_not_certified=True,training_admitted=False,goal_achieved=False))
    try:
        counts = Counter();native = defaultdict(list)
        with METADATA.open('rb') as stream:
            for line in stream:
                if b'"source_group": "govinfo:' in line:
                    item = json.loads(line);native[item['source_id']].append(item)
        if len(native) != 1382 or any(len(v) != 1 for v in native.values()):raise ValueError('All1382 complete prepared native views required')
        matched_vetoes = [];output = OUT/'all1433-fresh-requested-pixel-observations.jsonl';tasks = Counter()
        with (DATA/'train.jsonl').open('rb') as originals,(SOURCE/'all1433-original-train-label-observations.jsonl').open() as observed,output.open('x') as destination:
            for position,raw in enumerate(originals):
                source = json.loads(raw);item = json.loads(observed.readline())
                if (source['split'] != 'train' or item['original_source_role'] != 'train' or item['original_JSONL_position'] != position
                        or item['original_line_sha256'] != hashlib.sha256(raw).hexdigest() or item['source_id'] != source['id']
                        or item['task'] != source['task'] or item['original_response'] != source['response']
                        or item['fresh_original_PDF_response'] != source['response'] or item['prepared_native_views'] != native[source['id']]):
                    raise ValueError('Every complete original role/position/row/response and native source reference must reconstruct')
                required = {'original_full_PDF_target_matches','original_source_box_matches','original_parent_RGB224_hash_matches',
                            'every_saved_source_RGB_pixel_matches','original_saved_RGB224_hash_matches'}
                if set(item['source_reconstruction']) != required or not all(item['source_reconstruction'].values()):
                    raise ValueError('All original full-source reconstruction observations must remain present and successful')
                image_path = (DATA/source['image']).resolve(strict=True)
                if not image_path.is_relative_to(DATA) or sha(image_path) != source['stored_image_sha256']:
                    raise ValueError('Every actual original train source image byte must remain exact')
                with Image.open(image_path) as opened:
                    if list(opened.size) != item['actual_source_dimensions'] or min(opened.size) < 1 or max(opened.size) > 768:
                        raise ValueError('Every original bounded image geometry must match before decoding')
                    image = opened.convert('RGB')
                if source['task'] == 'document_ocr':box = [0,0,image.width,image.height]
                elif source['task'] == 'document_region_ocr':
                    l,t,r,b = source['normalized_box']
                    if not all(math.isfinite(v) for v in [l,t,r,b]) or not 0 <= l < r <= 1 or not 0 <= t < b <= 1:
                        raise ValueError('Whole original normalized requested box required')
                    box = [math.floor(l*image.width),math.floor(t*image.height),math.ceil(r*image.width),math.ceil(b*image.height)]
                else:raise ValueError('Only the original document training task families may be read')
                region = image.crop(box)
                # This is a direct scan of decoded actual pixels, independent
                # of the producer's getextrema/classifier helper.
                white = all(pixel == (255,255,255) for pixel in region.getdata())
                veto = white and bool(source['response'].strip())
                padded = ImageOps.pad(image,(224,224),method=Image.Resampling.BICUBIC,color=(128,128,128))
                if hashlib.sha256(padded.tobytes()).hexdigest() != source['model_pixel_sha256']:
                    raise ValueError('Every original padded uint8RGB224 hash must independently reproduce')
                reported = item['requested_source_pixels']
                if (list(image.size) != item['actual_source_dimensions'] or box != reported['original_requested_pixel_box']
                        or list(region.size) != reported['actual_requested_dimensions']
                        or hashlib.sha256(region.tobytes()).hexdigest() != reported['actual_requested_RGB_sha256']
                        or white != reported['actual_requested_region_all_white']
                        or veto != item['veto_unchanged_positive_for_future_admission']):
                    raise ValueError('Every complete original requested pixel geometry, RGB value and veto decision must independently match')
                counts['original_rows'] += 1;counts['prepared_native_views'] += len(native[source['id']]);tasks[source['task']] += 1
                counts['confirmed_white_requested_region_vetoes' if veto else 'not_vetoed_by_this_narrow_rule'] += 1
                if veto:matched_vetoes.append(dict(source_id=source['id'],original_JSONL_position=position,
                    original_line_sha256=hashlib.sha256(raw).hexdigest(),prepared_native_views=native[source['id']]))
                destination.write(json.dumps(dict(original_JSONL_position=position,source_id=source['id'],source_role='train',
                    whole_original_image_SHA256=source['stored_image_sha256'],actual_requested_box=box,actual_requested_dimensions=list(region.size),
                    actual_requested_RGB_sha256=hashlib.sha256(region.tobytes()).hexdigest(),every_requested_pixel_is_white=white,
                    independently_reproduced_veto=veto,prepared_native_views=len(native[source['id']]),training_admitted=False))+'\n')
                if (position+1)%64 == 0:
                    destination.flush();checked();atomic(OUT/'progress.json',dict(owner=owner,actual_counts=dict(counts),training_admitted=False))
            if observed.readline():raise ValueError('No original observation tail may be omitted')
            destination.flush();os.fsync(destination.fileno())
        if (counts['original_rows'] != 1433 or counts['prepared_native_views'] != 1382
                or {r['source_id'] for r in matched_vetoes} != set(proof['actual_confirmed_semantic_veto_source_ids'])
                or matched_vetoes != vetoes['unchanged_positive_source_vetoes']):
            raise ValueError('Every source/view and complete authoritative veto list must independently reproduce')
        checked()
        if any(sha(p) != digest for p,digest in files.items()):raise ValueError('Every frozen original image/PDF/source/runtime byte must remain exact')
        atomic(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
            actual_counts=dict(counts),actual_original_task_counts=dict(tasks),actual_all1433_requested_regions_directly_scanned=True,
            actual_confirmed_veto_source_ids=[r['source_id'] for r in matched_vetoes],actual_veto_list_matches_source_audit=True,
            all_original1382_native_views_accounted_for=True,excluded51_originals_not_readmitted=True,
            fresh_observations_sha256=sha(output),PDFs_not_all_rerendered_in_this_reader=True,
            every_nonvetoed_transcript_semantics_still_uncertified=True,new_qualified_correction_overlay_still_required=True,
            no_existing_source_or_training_recipe_or_neural_worker_changed=True,
            training_admitted=False,neural_forwards=0,new_production_optimizer_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=True,actual_counts=dict(counts),training_admitted=False)),flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json',dict(owner=owner,error=repr(error),automatic_retry=False,training_admitted=False,goal_achieved=False));raise


if __name__ == '__main__':main()
