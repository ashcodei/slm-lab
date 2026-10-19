"""Independently rebuild the retained24document source preflight; no admission.

Reads original pinned Parquet rows, whole own BPE/masks, license metadata and
the observed prefix disposition ledger. Does not import the intake selector.
"""
import ast
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote,urlsplit

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/public-pretraining-source-preflight-CPU-v1'
DATA=ROOT/'data/public-pretraining-source-preflight-v1'
OUT=ROOT/'reports/public-pretraining-source-preflight-CPU-v1-readback'


def verify_document(tokenizer,overlap,source,original,entry):
    family=source['family'];text=original['code' if family=='code' else 'text'];raw=text.encode('utf-8')
    if not 32<=len(raw)<=1024**2 or '\0' in text:raise ValueError('Every complete original bounded public UTF8 body is required')
    if entry['original_publisher_row']!=json.loads(json.dumps(original,default=str)) or entry['original_public_text']!=text:
        raise ValueError('Every original publisher field and complete source body must be retained')
    sha=hashlib.sha256(raw).hexdigest();values=tokenizer.encode(text,add_special_tokens=False).ids;ids=[1,*values,2]
    if (entry['original_UTF8_sha256']!=sha or entry['id']!='public-source:'+sha or entry['whole_native_ids']!=ids
            or entry['whole_native_labels']!=[-100,*ids[1:]] or entry['complete_context_tokens']!=len(ids)
            or entry['complete_causal_targets']!=len(ids)-1 or any(not 4<=v<32768 for v in values)
            or tokenizer.decode(values,skip_special_tokens=False)!=text):
        raise ValueError('Every whole original source byte, actual own BPE token, causal mask and EOS target must rebuild')
    if (entry['role']!=('valid' if int(sha[-8:],16)%100==0 else 'train')
            or entry['optimization_admitted'] is not False or entry['human_preferences'] is not False
            or entry['execution_reward'] is not False
            or any(entry[k]!=source[v] for k,v in [('family','family'),('source_repository','repo'),('source_revision','revision'),('source_parquet','path')])):
        raise ValueError('Exact public source, fixed role and explicit nonhuman unadmitted causal-data scope required')
    if family=='code':
        allowed={'mit','apache-2.0','bsd-2-clause','bsd-3-clause','isc','cc0-1.0','unlicense','zlib','0bsd'}
        if original.get('license','').casefold() not in allowed or original['language'] not in {'Python','C','C++','Java','JavaScript','TypeScript','Go','Rust'}:
            raise ValueError('Publisher per-file license and declared language must qualify independently of the catalog license')
        if original['language']=='Python':ast.parse(text)
        expected='https://github.com/'+original['repo_name']+'/blob/HEAD/'+quote(original['path'].lstrip('/'),safe='/')
        if (entry['original_public_URL']!=expected or entry['rights']['publisher_per_file_license']!=original['license']
                or not entry['rights']['dataset_catalog_license_does_not_replace_per_file_license']):
            raise ValueError('Original repository/path attribution and separate per-file license must remain exact')
    else:
        if (entry['original_public_URL']!=original['url'] or urlsplit(original['url']).scheme not in {'http','https'}
                or entry['rights']['dataset_license']!=source['dataset_license'] or not entry['rights']['underlying_page_rights_not_relicensed']):
            raise ValueError('Original public page URL and non-relicensed underlying-rights scope required')
    if overlap.literal(text) or overlap.lexical_copies(text):raise ValueError('Every actual complete native body must remain screened against the declared unchanged held stems')
    return dict(sha256=sha,targets=len(ids)-1,context_tokens=len(ids),role=entry['role'],family=family,language=entry['language'])


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh GPU-hidden original-public-source reader required')
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.held_text_overlap_v1 import HeldTextOverlap
    from slm_lab.local_source_closure_v2 import python_sources
    done=json.loads((SOURCE/'complete.json').read_text());plan=json.loads((SOURCE/'plan.json').read_text())
    if not terminal(plan['owner']) or not done['complete'] or done['actual_complete_original_documents']!=24 or done['optimization_admitted'] is not False:
        raise ValueError('The exact original public preflight must finish without training admission')
    tests=ROOT/'reports/public-pretraining-preflight-reader-v1-tests.log'
    if '2 passed' not in tests.read_text():raise ValueError('Actual full-source/mask corruption and per-file license controls required')
    files=dict(plan['source_files_sha256']);files.update(done['source_files_sha256'])
    files.update({str(p):file_hash(p) for p in [tests,*python_sources(ROOT,[Path(__file__),ROOT/'tests/test_public_pretraining_preflight_reader_v1.py'])]})
    registry=ROOT/'reports/CodeContests-native-held-text-v2/unchanged-held-text-registry.json';rows=json.loads(registry.read_text())
    if len(rows)!=32953:raise ValueError('Every original held stem required')
    overlap=HeldTextOverlap(rows);tokenizer=Tokenizer.from_file(str(ROOT/'data/tokenizer-corpus/tokenizer.json'));seen=set();proofs=[]
    for source in plan['pinned_sources']:
        folder=DATA/source['family'];report=json.loads((folder/'complete.json').read_text());files.update(report['source_files_sha256'])
        path=folder/'original.parquet'
        if file_hash(path)!=source['sha256'] or path.stat().st_size!=source['size']:raise ValueError('Actual complete public data file must match the pinned publisher LFS digest')
        entries=[json.loads(line) for line in (folder/'complete-original-native-documents.jsonl').open()]
        ledger=[json.loads(line) for line in (folder/'observed-prefix-dispositions.jsonl').open()]
        by_position={r['original_parquet_row_ordinal']:r for r in entries};observed=[]
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128,use_threads=False):
            observed.extend(batch.to_pylist())
            if len(observed)>=len(ledger):break
        original=observed[:len(ledger)]
        if len(original)!=len(ledger) or len(entries)!=8 or [r['original_row_ordinal'] for r in ledger]!=list(range(len(ledger))):
            raise ValueError('Every declared original prefix position and exactly eight complete source samples must be present')
        total=Counter()
        for p,(row,disposition) in enumerate(zip(original,ledger)):
            if p in by_position:
                entry=by_position[p];proof=verify_document(tokenizer,overlap,source,row,entry)
                if proof['sha256'] in seen or disposition['accepted_id']!=entry['id']:raise ValueError('Every original accepted sample must occur once with its exact source ledger position')
                seen.add(proof['sha256']);total.update(documents=1,targets=proof['targets'],context_tokens=proof['context_tokens'])
            else:
                if disposition['accepted_id'] is not None:raise ValueError('An excluded original row cannot become an accepted sample')
                reason=disposition['disposition']
                if reason=='per_file_license_not_in_declared_allowlist':
                    if row.get('license','').casefold() in {'mit','apache-2.0','bsd-2-clause','bsd-3-clause','isc','cc0-1.0','unlicense','zlib','0bsd'}:
                        raise ValueError('The retained excluded license qualification differs')
                elif reason=='outside_declared_programming_languages':
                    if row.get('language') in {'Python','C','C++','Java','JavaScript','TypeScript','Go','Rust'}:raise ValueError('The retained excluded language qualification differs')
                else:raise ValueError('This source reader does not admit an unverified new exclusion kind')
        if total['targets']!=report['actual_complete_native_causal_targets'] or len(ledger)!=report['actual_original_prefix_rows_observed']:
            raise ValueError('Every actual source-prefix and whole-native target count must rederive')
        proofs.append(dict(family=source['family'],actual_original_prefix_rows=len(ledger),**dict(total)))
    guard=FileBindings(files);guard.check(full=False);OUT.mkdir()
    atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,actual_source_prefix_and_native_proofs=proofs,
        actual_complete_original_documents=len(seen),actual_complete_native_causal_targets=sum(p['targets'] for p in proofs),
        all_whole_original_Parquet_rows_BPE_masks_EOS_rights_and_held_text_checks_rebuilt=True,
        selector_module_not_imported=True,optimization_admitted=False,human_preferences=False,
        this_is_a24document_preflight_not_billion_token_pretraining=True,
        larger_source_collection_independent_full_readback_and_training_still_required=True,
        source_files_sha256={**files,str(SOURCE/'complete.json'):file_hash(SOURCE/'complete.json')},
        model_updates=0,candidate_promoted=False,goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_original_documents=len(seen),optimization_admitted=False)),flush=True)


if __name__=='__main__':main()
