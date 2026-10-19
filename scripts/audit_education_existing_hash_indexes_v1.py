"""Limited exact/normalized educational overlap audit against retained indexes.

Read-only old hash indexes do not cover every earlier training source or near
duplicates. Absence of an indexed match is not a novelty/independence claim.
"""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'data/open-education-bookpack-v1/cross-corpus-hash-index-review.json'


def main():
    if OUTPUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Fresh GPU-hidden read-only index audit required')
    paths=[Path(p) for p in subprocess.check_output(['rg','--files','data','-g','*.sqlite'],cwd=ROOT,text=True).splitlines()]
    indexes=[];skipped=[]
    for relative in sorted(paths):
        path=(ROOT/relative).resolve();db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
        names={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'hashes' in names:
            query='SELECT id FROM hashes WHERE id=?';kind='casefold_and_collapsed_whitespace_full_text_SHA256_hex'
        elif 'documents' in names:
            query='SELECT part FROM documents WHERE hash=?';kind='collapsed_whitespace_Wikipedia_body_SHA256_bytes'
        elif 'code' in names:
            query='SELECT part,split FROM code WHERE hash=?';kind='stripped_original_code_SHA256_bytes'
        else:
            skipped.append(dict(path=str(relative),tables=sorted(names)));db.close();continue
        if any(Path(str(path)+suffix).exists() for suffix in ('-wal','-journal')):
            raise ValueError('An index with a live/pending journal requires separate inspection before this static audit')
        indexes.append(dict(path=path,connection=db,query=query,kind=kind,before_sha256=file_hash(path)))
    matches=[];counts=Counter();pages=0
    blob=ROOT/'data/open-education-source-v1/libretexts-0000.json.gz'
    source_sha=file_hash(blob)
    try:
        with gzip.open(blob,'rt') as stream:
            for position,line in enumerate(stream):
                row=json.loads(line);text=row['text'];pages+=1
                variants={'complete_original_page':text,'original_page_after_first_title_line':text.partition('\n')[2]}
                if '\n\n' in text:variants['original_page_after_first_blank_line']=text.partition('\n\n')[2]
                for index in indexes:
                    keys={}
                    for name,body in variants.items():
                        if not body.strip():continue
                        if index['kind'].startswith('casefold'):
                            key=hashlib.sha256(' '.join(body.casefold().split()).encode()).hexdigest()
                        elif index['kind'].startswith('collapsed'):
                            key=hashlib.sha256(' '.join(body.split()).encode()).digest()
                        else:key=hashlib.sha256(body.strip().encode()).digest()
                        keys.setdefault(key,[]).append(name)
                    for key,names in keys.items():
                        hit=index['connection'].execute(index['query'],(key,)).fetchone()
                        if hit:
                            counts[str(index['path'].relative_to(ROOT))]+=1
                            matches.append(dict(original_source_position=position,id=row['id'],book_url=row['metadata']['book_url'],
                                matched_retained_index=str(index['path'].relative_to(ROOT)),hash_method=index['kind'],variants=names,
                                retained_index_values=list(hit),prior_split_known=index['kind'].startswith('stripped')))
    finally:
        for index in indexes:index['connection'].close()
    for index in indexes:
        if file_hash(index['path'])!=index['before_sha256']:raise ValueError('A source index changed during read-only audit')
    if pages!=40049 or file_hash(blob)!=source_sha:raise ValueError('Every original unchanged educational row required')
    report=dict(completed_at=datetime.now(timezone.utc).isoformat(),script_sha256=file_hash(Path(__file__).resolve()),
        original_educational_source_sha256=source_sha,original_pages_scanned=pages,
        retained_indexes=[dict(path=str(r['path'].relative_to(ROOT)),sha256=r['before_sha256'],method=r['kind']) for r in indexes],
        skipped_index_schemas=skipped,matches=matches,matches_by_index=dict(counts),
        read_only_old_index_connections=True,scope='Only declared whole-text/body/code hash variants in the located retained compatible hash indexes.',
        all_previous_sources_or_substrings_or_semantic_near_duplicates_not_covered=True,
        training_novelty_or_semantic_independence_not_certified=True,training_admitted=False,model_updates_by_this_audit=0,goal_achieved=False)
    atomic_json(OUTPUT,report);print(json.dumps({k:report[k] for k in ['original_pages_scanned','retained_indexes','matches_by_index','training_admitted']},indent=2),flush=True)


if __name__=='__main__':main()
