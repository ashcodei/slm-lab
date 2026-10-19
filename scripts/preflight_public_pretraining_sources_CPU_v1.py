"""Fetch pinned public data files and inspect whole own-native source samples.

This is a three-source, eight-document-per-source transport/schema/licensing/
held-text/BPE preflight. It is not a billion-token corpus or training admission.
Every downloaded Parquet is retained with its actual publisher LFS digest.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from urllib.parse import quote
from urllib.request import Request,urlopen

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/public-pretraining-source-preflight-CPU-v1'
DATA=ROOT/'data/public-pretraining-source-preflight-v1'
RESERVE=40*1024**3
SOURCES=[
 dict(family='code',repo='codeparrot/github-code-clean',revision='c48d40f9e70f0196f8236901ee35807f7d6c44c0',
      path='data/train-00000-of-00880.parquet',size=355122552,sha256='c05e4cad93aac3f978a2385f8c055a10d247b107afd8bf9cb8c762f8403bdc0f',dataset_license='apache-2.0_catalog_per_file_licenses_separate'),
 dict(family='math',repo='HuggingFaceTB/finemath',revision='e92b25a616738fe95dc186b64dfb19f9c8525594',
      path='finemath-4plus/train-00000-of-00064.parquet',size=286267316,sha256='5d0b26114b4cf309c82a4ba8f6f45857480b27812e23a46b61b45e8bbed61fe5',dataset_license='odc-by'),
 dict(family='web',repo='HuggingFaceFW/fineweb-edu',revision='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9',
      path='sample/10BT/000_00000.parquet',size=2152819114,sha256='b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871',dataset_license='odc-by')]


def get(url):
    return urlopen(Request(url,headers={'User-Agent':'slm-lab-public-source-preflight/1.0'}),timeout=60)


def fetch(source):
    folder=DATA/source['family'];folder.mkdir();base='https://huggingface.co/datasets/'+source['repo']+'/resolve/'+source['revision']+'/'
    notice=folder/'publisher-README.md'
    with get(base+'README.md') as response:notice.write_bytes(response.read(4*1024**2))
    url=base+quote(source['path'],safe='/')+'?download=true'
    path=folder/'original.parquet';pending=folder/'original.parquet.pending';sha=hashlib.sha256();written=0;last=time.monotonic()
    if shutil.disk_usage(ROOT).free<RESERVE+source['size']:raise ValueError('Complete pinned source file must fit above the persistent reserve')
    with get(url) as response,pending.open('xb') as stream:
        while True:
            chunk=response.read(1024**2)
            if not chunk:break
            written+=len(chunk)
            if written>source['size'] or shutil.disk_usage(ROOT).free-len(chunk)<RESERVE:
                raise ValueError('Public source exceeded its pinned size or persistent reserve')
            sha.update(chunk);stream.write(chunk)
            if time.monotonic()-last>=10:
                atomic_json(OUT/'progress.json',dict(family=source['family'],downloaded_bytes=written,pinned_bytes=source['size'],model_updates=0,goal_achieved=False));last=time.monotonic()
        stream.flush();os.fsync(stream.fileno())
    if written!=source['size'] or sha.hexdigest()!=source['sha256']:raise ValueError('Actual complete public Parquet bytes must match the pinned publisher LFS identity')
    os.replace(pending,path)
    return path,notice


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or DATA.exists():raise ValueError('Fresh GPU-hidden original public-data preflight required')
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    from slm_lab.held_text_overlap_v1 import HeldTextOverlap
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.public_pretraining_intake_v1 import candidate
    tests=ROOT/'reports/public-pretraining-intake-v1-tests.log'
    if '4 passed' not in tests.read_text():raise ValueError('Actual source/mask/license/held-text preflight tests must pass')
    registry_path=ROOT/'reports/CodeContests-native-held-text-v2/unchanged-held-text-registry.json'
    registry=json.loads(registry_path.read_text())
    if len(registry)!=32953:raise ValueError('All original unchanged held stems required')
    token_path=ROOT/'data/tokenizer-corpus/tokenizer.json';tokenizer=Tokenizer.from_file(str(token_path))
    if tokenizer.get_vocab_size()!=32768:raise ValueError('Only the exact own native tokenizer is allowed')
    files={str(p):file_hash(p) for p in [tests,registry_path,token_path,*python_sources(ROOT,[Path(__file__),ROOT/'tests/test_public_pretraining_intake_v1.py'])]}
    guard=FileBindings(files);overlap=HeldTextOverlap(registry);OUT.mkdir();DATA.mkdir();started=time.monotonic()
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,pinned_sources=SOURCES,
        accepted_complete_documents_each_source=8,maximum_observed_original_rows_each_source=5000,
        held_overlap_configuration=overlap.configuration,whole_original_text_and_native_EOS_preserved=True,
        no_model_optimizer_GPU_or_external_teacher_weights_loaded=True,no_training_admission=True,
        larger_corpus_still_requires_separate_full_preparation_and_independent_readback=True,model_updates=0,goal_achieved=False))
    results=[];seen=set()
    try:
        for source in SOURCES:
            guard.check();path,notice=fetch(source);parquet=pq.ParquetFile(path);folder=path.parent
            atomic_json(folder/'original-schema.json',dict(schema=str(parquet.schema_arrow),original_file_rows=parquet.metadata.num_rows))
            accepted=[];observed=0;reasons=Counter();native_targets=0
            with (folder/'complete-original-native-documents.jsonl').open('x') as docs,(folder/'observed-prefix-dispositions.jsonl').open('x') as ledger:
                for batch in parquet.iter_batches(batch_size=128,use_threads=False):
                    for row in batch.to_pylist():
                        original_ordinal=observed;observed+=1
                        item,reason=candidate(row,source,tokenizer,overlap)
                        if item is not None and item['original_UTF8_sha256'] in seen:item=None;reason='byte_identical_document_already_in_preflight'
                        if item is not None:
                            seen.add(item['original_UTF8_sha256']);item['original_parquet_row_ordinal']=original_ordinal
                            docs.write(json.dumps(item,ensure_ascii=False,default=str)+'\n');docs.flush();accepted.append(item['id']);native_targets+=item['complete_causal_targets']
                        reasons[reason]+=1;ledger.write(json.dumps(dict(original_row_ordinal=original_ordinal,disposition=reason,accepted_id=None if item is None else item['id']))+'\n')
                        if len(accepted)==8 or observed>=5000:break
                    if len(accepted)==8 or observed>=5000:break
                os.fsync(docs.fileno());os.fsync(ledger.fileno())
            if len(accepted)!=8:raise ValueError('Every declared public source must yield eight complete independently inspectable native documents')
            evidence=[path,notice,folder/'original-schema.json',folder/'complete-original-native-documents.jsonl',folder/'observed-prefix-dispositions.jsonl']
            result=dict(family=source['family'],source=source,actual_source_file_SHA256=file_hash(path),actual_original_file_bytes=path.stat().st_size,
                original_file_total_rows=parquet.metadata.num_rows,actual_original_prefix_rows_observed=observed,actual_accepted_documents=8,
                actual_complete_native_causal_targets=native_targets,actual_prefix_dispositions=dict(reasons),
                original_publisher_rows_and_whole_own_BPE_responses_retained=True,optimization_admitted=False,
                source_files_sha256={str(p):file_hash(p) for p in evidence},goal_achieved=False)
            atomic_json(folder/'complete.json',result);results.append(result)
            atomic_json(OUT/'source-progress.json',dict(complete_source_families=[r['family'] for r in results],model_updates=0,goal_achieved=False))
            print(json.dumps(dict(family=source['family'],whole_public_documents=8,observed_original_rows=observed,native_targets=native_targets)),flush=True)
        guard.check(full=True)
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,sources=results,
            actual_complete_original_documents=24,actual_complete_native_causal_targets=sum(r['actual_complete_native_causal_targets'] for r in results),
            original_public_data_files_byte_verified_against_publisher_LFS=True,
            original_code_file_licenses_separate_from_dataset_catalog=True,all32953declared_held_stems_screen_actual_native_text=True,
            new_neural_training_or_quality_gain=False,optimization_admitted=False,
            not_a_completed_billion_token_corpus=True,semantic_and_all_history_independence_not_certified=True,
            source_files_sha256={str(OUT/'plan.json'):file_hash(OUT/'plan.json'),**{str(DATA/r['family']/'complete.json'):file_hash(DATA/r['family']/'complete.json') for r in results}},
            seconds=time.monotonic()-started,model_updates=0,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),original_public_documents=24,optimization_admitted=False)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),completed_source_families=[r['family'] for r in results],
            partial_source_files_preserved=True,original_GPU_training_not_signalled=True,optimization_admitted=False,goal_achieved=False));raise


if __name__=='__main__':main()
