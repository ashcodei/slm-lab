"""Collect one billion whole public causal training targets as an unadmitted candidate.

Code500M across eight languages, math250M and educational web250M. All complete
source documents and own native IDs are retained, while new transient download
files can be removed after their hashes and accepted source rows are durable.
This does not alter, start, stop or own any neural/GPU training process.
"""
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import traceback
from urllib.request import Request,urlopen

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/public-billion-causal-candidate-CPU-v2'
DATA=ROOT/'data/public-billion-causal-candidate-v2'
PREFLIGHT=ROOT/'reports/public-pretraining-source-preflight-CPU-v1'
PILOT_DATA=ROOT/'data/public-pretraining-source-preflight-v1'
RESERVE=40*1024**3
MAX_ARTIFACTS=12*1024**3


def catalog(source):
    directory=source['path'].rsplit('/',1)[0]
    url='https://huggingface.co/api/datasets/'+source['repo']+'/tree/'+source['revision']+'/'+directory+'?limit=1000'
    rows=[]
    for _ in range(100):
        with urlopen(Request(url,headers={'User-Agent':'slm-lab-public-corpus-candidate/1.0'}),timeout=60) as response:
            rows.extend(json.load(response));link=response.headers.get('Link','')
        match=re.search(r'<([^>]+)>;\s*rel="next"',link)
        if not match:break
        url=match[1]
        if not url.startswith('https://huggingface.co/api/datasets/'+source['repo']+'/tree/'+source['revision']+'/'):
            raise ValueError('Every immutable source catalog page must retain the pinned public repository revision')
    else:raise ValueError('Bounded complete pinned source catalog required')
    files=[]
    for row in sorted(rows,key=lambda r:r['path']):
        if row['type']!='file' or not row['path'].endswith('.parquet'):continue
        lfs=row.get('lfs',{});sha=lfs.get('oid');size=row.get('size')
        if not isinstance(sha,str) or len(sha)!=64 or type(size) is not int or size<1 or lfs.get('size')!=size:
            raise ValueError('Every complete source file needs its original public LFS digest and byte size')
        files.append(dict(source,path=row['path'],size=size,sha256=sha))
    if not files or files[0]['sha256']!=source['sha256'] or files[0]['path']!=source['path']:
        raise ValueError('Full public source order must extend the exact independently checked first source file')
    return files


def owned_bytes():return sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file())


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or REPORTS.exists() or DATA.exists():
        raise ValueError('Fresh GPU-hidden whole-public-corpus candidate required')
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    import preflight_public_pretraining_sources_CPU_v1 as transport
    from slm_lab.held_text_overlap_v1 import HeldTextOverlap
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.public_billion_candidate_v1 import CandidateWriter,QUOTAS,key
    from slm_lab.public_pretraining_intake_v2 import candidate,language as canonical_language
    reader_path=ROOT/'reports/public-pretraining-source-preflight-CPU-v1-readback/complete.json';reader=json.loads(reader_path.read_text())
    if (not reader['complete'] or reader['actual_complete_original_documents']!=24 or reader['optimization_admitted'] is not False
            or not reader['all_whole_original_Parquet_rows_BPE_masks_EOS_rights_and_held_text_checks_rebuilt']):
        raise ValueError('The exact original three-source independent preflight must finish first')
    tests={'public-pretraining-intake-v1-tests':4,'public-pretraining-preflight-reader-v1-tests':2,
           'public-billion-candidate-v1-tests':3,'public-billion-source-driver-v1-tests':2,'public-pretraining-intake-v2-tests':3}
    files=dict(reader['source_files_sha256']);files[str(reader_path)]=file_hash(reader_path)
    for name,count in tests.items():
        path=ROOT/'reports'/(name+'.log')
        if str(count)+' passed' not in path.read_text():raise ValueError('Actual whole-source/mask/license/dedup/quota controls must pass before collection')
        files[str(path)]=file_hash(path)
    files.update({str(p):file_hash(p) for p in python_sources(ROOT,[Path(__file__),ROOT/'tests/test_public_billion_candidate_v1.py'])})
    guard=FileBindings(files);registry_path=ROOT/'reports/CodeContests-native-held-text-v2/unchanged-held-text-registry.json'
    registry=json.loads(registry_path.read_text());overlap=HeldTextOverlap(registry)
    if len(registry)!=32953:raise ValueError('Every original unchanged held stem must remain in source screening')
    tokenizer=Tokenizer.from_file(str(ROOT/'data/tokenizer-corpus/tokenizer.json'))
    source_plan=json.loads((PREFLIGHT/'plan.json').read_text());catalogs={s['family']:catalog(s) for s in source_plan['pinned_sources']}
    if shutil.disk_usage(ROOT).free<RESERVE+MAX_ARTIFACTS:raise ValueError('The complete candidate bound and persistent reserve must fit')
    REPORTS.mkdir();DATA.mkdir();(REPORTS/'transport').mkdir();temp=DATA/'transient-public-downloads';temp.mkdir()
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(REPORTS/'source-catalogs.json',catalogs)
    atomic_json(REPORTS/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        source_catalog_sha256=file_hash(REPORTS/'source-catalogs.json'),fixed_train_target_quotas=QUOTAS,
        minimum_complete_training_targets=10**9,whole_last_documents_may_exceed_quotas_no_clipping=True,
        publisher_GO_to_Go_spelling_normalized_original_rows_and_native_bytes_preserved=True,
        exact_body_deduplication_across_this_new_candidate=True,prior_all_history_novelty_not_certified=True,
        role_rule='Exact original body SHA256 last32bits modulo100: zero is valid, all other values train',
        all32953declared_held_stems_and_lexical_copy_candidates_excluded=True,
        maximum_owned_artifact_bytes=MAX_ARTIFACTS,minimum_free_persistent_bytes=RESERVE,
        known_pinned_source_downloads_byte_verified_then_owned_transient_files_may_be_removed=True,
        full_independent_readback_still_required_before_any_training_admission=True,
        no_GPU_or_neural_weights_loaded=True,optimization_admitted=False,model_updates=0,goal_achieved=False))
    writer=CandidateWriter(DATA/'whole-native-documents',QUOTAS);counts=Counter();source_number=0;observed=0;started=time.monotonic();last=started
    transport.OUT=REPORTS/'transport';transport.RESERVE=RESERVE;deadline=time.monotonic()+72*3600
    ledger=gzip.open(DATA/'observed-original-prefix-dispositions.jsonl.gz','xt',encoding='utf-8')
    receipts=(DATA/'original-source-file-receipts.jsonl').open('x')
    def progress():
        writer.flush();ledger.flush();os.fsync(ledger.fileno());receipts.flush();os.fsync(receipts.fileno())
        size=owned_bytes();free=shutil.disk_usage(ROOT).free
        if size>MAX_ARTIFACTS or free<RESERVE:raise ValueError('Actual whole-source candidate or persistent reserve exceeded its declared bound')
        atomic_json(REPORTS/'progress.json',dict(at=datetime.now(timezone.utc).isoformat(),actual_source_files=source_number,
            original_rows_observed=observed,actual_prefix_dispositions=dict(counts),candidate=writer.summary(),
            actual_owned_artifact_bytes=size,actual_free_persistent_bytes=free,seconds=time.monotonic()-started,
            neural_training_updates=0,optimization_admitted=False,goal_achieved=False))
    try:
        for family in ['code','math','web']:
            for spec in catalogs[family]:
                if all(writer.full(k) for k in QUOTAS if k==family or k.startswith(family+':')):break
                guard.check();source_number+=1;work=None
                first=next(s for s in source_plan['pinned_sources'] if s['family']==family)
                reused=spec['path']==first['path'] and spec['sha256']==first['sha256']
                if reused:path=PILOT_DATA/family/'original.parquet'
                else:
                    work=temp/str(source_number);work.mkdir();transport.DATA=work
                    path,_=transport.fetch(spec)
                if file_hash(path)!=spec['sha256'] or path.stat().st_size!=spec['size']:
                    raise ValueError('Every actual complete original public source file must match its pinned LFS digest')
                parquet=pq.ParquetFile(path);ordinal=0;source_counts=Counter()
                for batch in parquet.iter_batches(batch_size=128,use_threads=False):
                    for row in batch.to_pylist():
                        if time.monotonic()>deadline:raise ValueError('Declared source collection lifetime ended; original training is untouched')
                        language=canonical_language(row.get('language'))
                        budget_key=key(family,language) if family!='code' or isinstance(language,str) else 'code:unsupported_publisher_language'
                        item=None
                        if writer.full(budget_key):reason='outside_declared_or_already_satisfied_training_quota'
                        else:
                            item,reason=candidate(row,spec,tokenizer,overlap)
                            if item is not None:
                                item['original_parquet_row_ordinal']=ordinal;item['original_source_file_sha256']=spec['sha256']
                                reason=writer.append(item)
                        ledger.write(json.dumps(dict(family=family,source_file=source_number,original_row_ordinal=ordinal,
                            disposition=reason,accepted_id=item['id'] if item is not None and reason=='whole_original_document_retained_unadmitted' else None),separators=(',',':'))+'\n')
                        ordinal+=1;observed+=1;counts[reason]+=1;source_counts[reason]+=1
                        if time.monotonic()-last>=10:progress();last=time.monotonic()
                        if all(writer.full(k) for k in QUOTAS if k==family or k.startswith(family+':')):break
                    if all(writer.full(k) for k in QUOTAS if k==family or k.startswith(family+':')):break
                receipts.write(json.dumps(dict(source_number=source_number,source=spec,actual_verified_original_file_bytes=path.stat().st_size,
                    actual_verified_original_file_sha256=spec['sha256'],original_file_total_rows=parquet.metadata.num_rows,
                    original_prefix_rows_observed=ordinal,source_prefix_dispositions=dict(source_counts),
                    full_pinned_preflight_file_reused=reused,nonpilot_transient_raw_file_not_retained_after_durable_source_capture=not reused),separators=(',',':'))+'\n')
                progress()
                if work is not None:
                    # Only this new producer's own transient download directory.
                    if work.parent!=temp or not work.name.isdecimal():raise ValueError('Refuse to remove any non-owned source path')
                    shutil.rmtree(work)
            if not all(writer.full(k) for k in QUOTAS if k==family or k.startswith(family+':')):
                raise ValueError('Pinned public source exhausted before its complete declared training quotas')
        if not writer.complete():raise ValueError('Every declared billion-target source quota must finish')
        summary=writer.summary();progress();writer.close();ledger.close();receipts.close();guard.check(full=True)
        artifacts=[p for p in DATA.rglob('*') if p.is_file()]
        atomic_json(DATA/'manifest.json',dict(format='own-whole-public-billion-causal-candidate-v1',**summary,
            fixed_train_target_quotas=QUOTAS,tokenizer_sha256=file_hash(ROOT/'data/tokenizer-corpus/tokenizer.json'),
            native_token_dtype='little_endian_uint16',whole_original_documents_retained=True,
            source_files_sha256={str(p):file_hash(p) for p in artifacts},
            independent_full_source_and_native_readback_required=True,optimization_admitted=False,human_preferences=False,execution_reward=False,goal_achieved=False))
        atomic_json(REPORTS/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,**summary,
            original_rows_observed=observed,actual_source_files=source_number,actual_prefix_dispositions=dict(counts),
            whole_native_targets_collected_not_neurally_trained=True,independent_full_readback_and_training_capacity_still_required=True,
            source_files_sha256={str(REPORTS/'plan.json'):file_hash(REPORTS/'plan.json'),str(REPORTS/'source-catalogs.json'):file_hash(REPORTS/'source-catalogs.json'),
                str(DATA/'manifest.json'):file_hash(DATA/'manifest.json')},seconds=time.monotonic()-started,
            neural_training_updates=0,optimization_admitted=False,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(REPORTS/'complete.json'),actual_train_targets=summary['actual_train_targets'],optimization_admitted=False)),flush=True)
    except BaseException as error:
        writer.close();ledger.close();receipts.close()
        atomic_json(REPORTS/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),partial_candidate=writer.summary(),
            original_training_not_signalled=True,partial_original_data_retained=True,optimization_admitted=False,goal_achieved=False));raise


if __name__=='__main__':main()
