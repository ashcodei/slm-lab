"""Audit an immutable committed prefix while the original collector keeps running.

No partial VQA training data is admitted, no publisher request is made and no
original source, journal, image or running process is changed.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.vqa_pair_admission import verified_photo
from slm_lab.vqa_pairs import canonical_question,known_photo_aliases

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/vqa-real-pairs-v1'
OUT=ROOT/'reports/committed-paired-photo-audit-v1'


def identity(pid):
    p=Path('/proc')/str(pid)
    try:
        fields=(p/'stat').read_text().split(') ',1)[1].split()
        return dict(pid=pid,start_ticks=fields[19],live=fields[0]!='Z',argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    except FileNotFoundError:return dict(pid=pid,live=False)


def main():
    if OUT.exists():raise ValueError('New distinct owned prefix audit required')
    actor=identity(130149)
    if not actor['live'] or actor.get('start_ticks')!='20300752' or 'slm_lab.vqa_pairs' not in actor['argv']:
        raise ValueError('Exact current original collector identity required')
    config_path=DATA/'configuration.json';config=json.loads(config_path.read_text());config_sha=file_hash(config_path)
    journal=DATA/'images.collected.jsonl';raw=journal.read_bytes();prefix=raw[:raw.rfind(b'\n')+1]
    rows=list(map(json.loads,prefix.splitlines()));by_id={(r['source_subset'],r['coco_image_id']):r for r in rows}
    if len(rows)!=len(by_id) or not rows:raise ValueError('Complete distinct committed original photo rows required')
    OUT.mkdir();(OUT/'committed-prefix.jsonl').write_bytes(prefix)
    def check(row):
        metadata=row['original_COCO_image_metadata']
        if (metadata['id']!=row['coco_image_id'] or metadata['license']!=4
                or known_photo_aliases(metadata)!={row['source_flickr_photo_id']}):
            raise ValueError('Exact original photo/COCO identity and declared source license required')
        verified_photo(DATA,row)
        return dict(image=str(DATA/row['image']),image_sha256=row['stored_image_sha256'],
            attribution=str(DATA/row['attribution_page']),attribution_sha256=row['attribution_page_sha256'])
    with ThreadPoolExecutor(max_workers=4) as pool:verified=list(pool.map(check,rows))
    candidates={};candidate_paths={};availability={};answer_types=Counter();binary_consensus={}
    for split in ['train','valid']:
        path=ROOT/'data/vqa-source-cache-v1'/('candidate-pairs.'+split+'.jsonl')
        digest=file_hash(path)
        if config['original_sources_sha256'].get(str(path))!=digest:raise ValueError('Original candidate question source differs')
        candidate_paths[str(path)]=digest;candidates[split]=list(map(json.loads,path.read_text().splitlines()))
        available=0;binary=0
        for pair in candidates[split]:
            canonical_question(pair)
            complete=all((pair['source_subset'],pair[side]['image']['id']) in by_id for side in ['first','second'])
            if not complete:continue
            for side in ['first','second']:
                sample=pair[side];record=by_id[(pair['source_subset'],sample['image']['id'])]
                if record['original_COCO_image_metadata']!=sample['image']:raise ValueError('Original question/photo metadata differs')
                answers=sample['annotation']['answers']
                if len(answers)!=10 or sorted(a['answer_id'] for a in answers)!=list(range(1,11)):raise ValueError('All original human votes required')
                if sum(a['answer']==sample['answer'] for a in answers)!=sample['exact_human_votes']:raise ValueError('Original human vote count differs')
                answer_types[sample['annotation']['answer_type']]+=1
            available+=1
            if ({pair[s]['answer'] for s in ['first','second']}=={'yes','no'}
                    and all(pair[s]['exact_human_votes']>=9 and pair[s]['annotation']['answer_type']=='yes/no' for s in ['first','second'])):
                binary+=1
        availability[split]=available;binary_consensus[split]=binary
    latest=journal.read_bytes()
    if not latest.startswith(prefix) or file_hash(config_path)!=config_sha:raise ValueError('Original prefix/configuration changed during readback')
    after=identity(130149)
    if after.get('live') and after.get('start_ticks')!=actor['start_ticks']:raise ValueError('Original collector identity changed')
    result=dict(observed_at=datetime.now(timezone.utc).isoformat(),passed=True,
        collector_at_start=actor,collector_after_readback=after,committed_photos=len(rows),
        current_committed_prefix_bytes=len(prefix),prefix_sha256=hashlib.sha256(prefix).hexdigest(),
        all_committed_source_attributions_and224_448px_pixels_verified=True,original_journal_remains_append_only=True,
        source_subset_counts=dict(Counter(r['source_subset'] for r in rows)),
        candidate_pairs_with_both_committed_photos_before_final_screens=availability,
        high_consensus_binary_candidate_pairs_before_final_screens=binary_consensus,
        candidate_answer_types=dict(answer_types),original_configuration_sha256=config_sha,
        candidate_files_sha256=candidate_paths,verified_original_files=verified,
        code_sha256={str(Path(__file__).resolve()):file_hash(__file__),
            str(ROOT/'src/slm_lab/vqa_pair_admission.py'):file_hash(ROOT/'src/slm_lab/vqa_pair_admission.py'),
            str(ROOT/'src/slm_lab/vqa_attribution.py'):file_hash(ROOT/'src/slm_lab/vqa_attribution.py')},
        full_dataset_finalized=(DATA/'manifest.json').exists(),training_admitted=False,model_updates=0,goal_achieved=False,
        limitations=['Availability counts precede official alias screening, validation-pair limits and final exact/perceptual protections.',
            'Stored JPEGs/pixels and retained current declarations are verified; original downloaded JPEG bytes are not retained for independent raw-hash reconstruction.',
            'This is a committed-prefix audit, not final paired-source admission, independent ownership adjudication or model quality.'])
    atomic_json(OUT/'results.json',result)
    print(json.dumps({k:result[k] for k in ['passed','committed_photos','source_subset_counts',
        'candidate_pairs_with_both_committed_photos_before_final_screens','high_consensus_binary_candidate_pairs_before_final_screens',
        'training_admitted','goal_achieved']},indent=2))


if __name__=='__main__':main()
