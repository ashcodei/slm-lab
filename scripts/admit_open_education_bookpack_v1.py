"""Review every retained publisher attribution field and admit original train text.

This operational admission preserves all original licence declarations and
missing metadata. It does not invent an author/version, classify everything
as public domain, or claim semantic novelty. No source text is republished.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/open-education-bookpack-v1'
FORMAT='whole-original-open-education-train-only-admission-v1'
DECLARATIONS={
    'Creative Commons - Attribution - https://creativecommons.org/licenses/by/4.0/',
    'Creative Commons - Attribution - https://creativecommons.org/licenses/by/3.0/',
    'Creative Commons - Attribution Share-Alike - https://creativecommons.org/licenses/by-sa/4.0/',
    'Creative Commons - Attribution Share-Alike - https://creativecommons.org/licenses/by-sa/3.0/',
    'Creative Commons - Attribution Share-Alike - https://creativecommons.org/licenses/by-sa/2.5/',
    'Public Domain','GNU Free Documentation License'}


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Source metadata review must explicitly hide GPUs')
    if any((DATA/n).exists() for n in ('publisher-attribution-review.json','admission.json')):
        raise ValueError('Preserve the first complete attribution/admission decision')
    manifest=json.loads((DATA/'corpus.json').read_text());raw=json.loads((DATA/'independent-raw-source-and-token-readback.json').read_text())
    overlap=json.loads((DATA/'cross-corpus-hash-index-review.json').read_text())
    files={**manifest['source_files_sha256'],str(DATA/'corpus.json'):file_hash(DATA/'corpus.json'),
        str(DATA/'original-pages.jsonl'):manifest['original_page_index_sha256'],
        str(DATA/'independent-raw-source-and-token-readback.json'):file_hash(DATA/'independent-raw-source-and-token-readback.json'),
        str(DATA/'cross-corpus-hash-index-review.json'):file_hash(DATA/'cross-corpus-hash-index-review.json'),str(Path(__file__).resolve()):file_hash(__file__)}
    guard=FileBindings(files)
    if (raw['source_manifest_sha256']!=file_hash(DATA/'corpus.json') or raw['all_source_metadata_and_raw_text_hashes_reconstructed']!=40049
            or not raw['every_packed_page_own_BPE_BOS_text_EOS_and_token_byte_reconstructed']
            or not raw['every_real_window_next_token_target_including_partial_tail_reconstructed']
            or raw['original_book_and_full_text_cross_role_overlap']!=0 or overlap['original_pages_scanned']!=40049
            or overlap['matches'] or not raw['named_held_book_components_not_packed']):
        raise ValueError('Complete actual original source/token/tail/held-role and retained-index evidence required')
    books=defaultdict(list);licenses=Counter();missing=Counter();packed=Counter();all_ids=set()
    for position,line in enumerate((DATA/'original-pages.jsonl').open()):
        row=json.loads(line);meta=row['metadata'];url=urlsplit(meta['url']);book=urlsplit(meta['book_url'])
        if (row['original_source_position']!=position or row['id'] in all_ids or not row['full_original_text_sha256']
                or meta['license'] not in DECLARATIONS or url.scheme!='https' or not (url.hostname or '').endswith('.libretexts.org')
                or book.scheme!='https' or book.hostname!='commons.libretexts.org' or not book.path.startswith('/book/')):
            raise ValueError('Every unique original source, declared publisher license and actual source/book URL required')
        all_ids.add(row['id']);licenses[meta['license']]+=1;books[meta['book_url']].append(row)
        expected=[]
        if not meta['author']:expected.append('publisher author missing')
        if not meta['title']:expected.append('publisher title missing')
        if meta['license']=='GNU Free Documentation License':expected.append('publisher GFDL version unrecorded')
        if row['attribution_review']!=expected:raise ValueError('Every original missing author/title/version flag must remain explicit')
        missing.update(expected)
        if row['native'] is not None:packed[row['native']['role']]+=1
    if (len(all_ids)!=40049 or len(books)!=782 or dict(licenses)!=manifest['original_license_counts']
            or dict(missing)!=manifest['unresolved_publisher_attribution_counts']
            or dict(packed)!={r:manifest['splits'][r]['documents'] for r in ('train','valid')}):
        raise ValueError('Every original page/book/metadata issue and packed role must reconstruct')
    ledger={}
    for book,rows in sorted(books.items()):
        roles={r['original_book_role'] for r in rows}
        if len(roles)!=1:raise ValueError('An original whole book cannot cross training and held roles')
        ledger[book]=dict(original_pages=len(rows),original_role=next(iter(roles)),
            packed_roles=dict(Counter(r['native']['role'] for r in rows if r['native'] is not None)),
            original_literal_author_values=dict(Counter(json.dumps(r['metadata']['author'],ensure_ascii=False,sort_keys=True) for r in rows)),
            literal_publisher_licenses=dict(Counter(r['metadata']['license'] for r in rows)),
            unresolved_publisher_metadata=dict(Counter(v for r in rows for v in r['attribution_review'])),
            original_page_positions=[r['original_source_position'] for r in rows])
    guard.check(full=True)
    review=dict(completed_at=datetime.now(timezone.utc).isoformat(),format='whole-education-original-publisher-attribution-review-v1',
        source_files_sha256=files,source_manifest_sha256=file_hash(DATA/'corpus.json'),
        complete_original_page_index_sha256=file_hash(DATA/'original-pages.jsonl'),every_original_page_reviewed=40049,
        every_original_book_reviewed=782,literal_publisher_license_counts=dict(licenses),
        unresolved_publisher_metadata_values_retained=dict(missing),no_invented_author_title_or_license_version=True,
        no_page_author_inferred_from_another_page=True,book_ledger=ledger,
        operational_review='Retain original public-source declarations, URLs, raw attribution fields and every unresolved value; authorize only the previously packed original train role.',
        declaration_review_does_not_certify_all_text_public_domain=True,GFDL_versions_remain_unrecorded=True,
        source_semantic_or_all_previous_corpus_independence_not_certified=True,
        no_source_redistribution_or_model_publication_by_this_action=True,model_updates=0,goal_achieved=False)
    atomic_json(DATA/'publisher-attribution-review.json',review)
    atomic_json(DATA/'admission.json',dict(format=FORMAT,admitted_at=datetime.now(timezone.utc).isoformat(),training_admitted=True,
        admitted_role='train',held_original_gradients_forbidden=True,
        corpus_sha256=file_hash(DATA/'corpus.json'),original_page_index_sha256=file_hash(DATA/'original-pages.jsonl'),
        full_raw_token_readback_sha256=file_hash(DATA/'independent-raw-source-and-token-readback.json'),
        retained_index_overlap_review_sha256=file_hash(DATA/'cross-corpus-hash-index-review.json'),
        publisher_attribution_review_sha256=file_hash(DATA/'publisher-attribution-review.json'),
        training_original_pages=manifest['splits']['train']['documents'],training_original_books=manifest['splits']['train']['original_books'],
        original_train_tokens=manifest['splits']['train']['tokens'],all_train_next_token_targets=manifest['splits']['train']['all_real_next_token_targets'],
        held_original_books=manifest['splits']['valid']['original_books'],unresolved_original_metadata_remains_explicit=True,
        all_material_public_domain_not_claimed=True,semantic_novelty_not_certified=True,preparation_model_updates=0,goal_achieved=False))
    print('admitted_complete_original_education',manifest['splits']['train']['documents'],'pages',manifest['splits']['train']['tokens'],'tokens')


if __name__=='__main__':main()
