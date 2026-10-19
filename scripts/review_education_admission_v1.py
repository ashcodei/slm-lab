"""Read original raw metadata and verify the full education admission ledger."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip
import json
import os
from pathlib import Path

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/open-education-bookpack-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Raw publisher metadata reader must explicitly hide GPUs')
    output=DATA/'admission-readback.json'
    if output.exists():raise ValueError('Preserve the original admission readback')
    admission=json.loads((DATA/'admission.json').read_text());review=json.loads((DATA/'publisher-attribution-review.json').read_text())
    corpus=json.loads((DATA/'corpus.json').read_text());rows=[json.loads(l) for l in (DATA/'original-pages.jsonl').open()]
    files={**review['source_files_sha256'],str(DATA/'admission.json'):file_hash(DATA/'admission.json'),str(DATA/'publisher-attribution-review.json'):file_hash(DATA/'publisher-attribution-review.json')}
    guard=FileBindings(files);books=defaultdict(list);missing=Counter();licenses=Counter();packed=Counter()
    with gzip.open(ROOT/'data/open-education-source-v1/libretexts-0000.json.gz','rt') as stream:
        for position,line in enumerate(stream):
            raw=json.loads(line);row=rows[position];meta=raw['metadata']
            if row['id']!=raw['id'] or row['metadata']!=meta or row['original_source_position']!=position:
                raise ValueError('Every actual original publisher metadata field/source identity required')
            books[meta['book_url']].append(row);licenses[meta['license']]+=1
            issues=[]
            if not meta['author']:issues.append('publisher author missing')
            if not meta['title']:issues.append('publisher title missing')
            if meta['license']=='GNU Free Documentation License':issues.append('publisher GFDL version unrecorded')
            if issues!=row['attribution_review']:raise ValueError('Every unresolved original value must remain literal')
            missing.update(issues)
            if row['native'] is not None:packed[row['native']['role']]+=1
    if (len(rows)!=40049 or len(books)!=782 or dict(missing)!=review['unresolved_publisher_metadata_values_retained']
            or dict(licenses)!=review['literal_publisher_license_counts'] or admission['corpus_sha256']!=file_hash(DATA/'corpus.json')
            or admission['publisher_attribution_review_sha256']!=file_hash(DATA/'publisher-attribution-review.json')
            or admission['training_admitted'] is not True or admission['held_original_gradients_forbidden'] is not True):
        raise ValueError('All original declarations, unresolved counts and actual train-only admission must reconcile')
    for book,records in books.items():
        expected=dict(original_pages=len(records),original_role=records[0]['original_book_role'],
            packed_roles=dict(Counter(r['native']['role'] for r in records if r['native'] is not None)),
            original_literal_author_values=dict(Counter(json.dumps(r['metadata']['author'],ensure_ascii=False,sort_keys=True) for r in records)),
            literal_publisher_licenses=dict(Counter(r['metadata']['license'] for r in records)),
            unresolved_publisher_metadata=dict(Counter(v for r in records for v in r['attribution_review'])),
            original_page_positions=[r['original_source_position'] for r in records])
        if review['book_ledger'].get(book)!=expected or len({r['original_book_role'] for r in records})!=1:
            raise ValueError('Every original whole-book ledger, literal author and held role must reconstruct')
    if dict(packed)!={r:corpus['splits'][r]['documents'] for r in ('train','valid')}:
        raise ValueError('Every actual retained training/held source count required')
    guard.check(full=True)
    atomic_json(output,dict(completed_at=datetime.now(timezone.utc).isoformat(),reader_sha256=file_hash(__file__),
        admission_sha256=file_hash(DATA/'admission.json'),publisher_review_sha256=file_hash(DATA/'publisher-attribution-review.json'),
        all40049_original_raw_page_metadata_and782whole_book_ledger_entries_reconstructed=True,
        original_unresolved_metadata_retained=dict(missing),original_publisher_license_counts=dict(licenses),
        no_invented_page_author_title_or_license_version=True,training_pages=packed['train'],held_pages=packed['valid'],
        admission_train_only_and_all_held_books_outside_gradients=True,semantic_novelty_not_certified=True,
        earlier_complete_raw_text_BPE_and_tail_readback_sha256=file_hash(DATA/'independent-raw-source-and-token-readback.json'),
        actual_model_updates=0,goal_achieved=False))
    print(output)


if __name__=='__main__':main()
