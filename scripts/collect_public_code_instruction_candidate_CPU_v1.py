"""Collect one pinned public dataset shard; no model weights or code execution.

This is synthetic teacher-produced candidate data, never human feedback or
on-policy RL. Collection alone does not admit any sample for optimization.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/public-code-instruction-collection-CPU-v1'
DATA = ROOT / 'data/public-code-instruction-candidate-v1'
REVISION = '8f3ba5bafe4d6e8db46082cf7ae6741bc370604d'
FILE = 'data/train-00000-of-00050.parquet'
BYTES = 133476478
SHA = '342757e0c6b706c8f68cf0a867ead5c417d0273726453dbcfc934f5c3c6ca891'
PREFIX = 1024
RESERVE = 40 * 1024**3
LIMIT = 512 * 1024**2


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=True) + '\n')


def download(url, path, maximum):
    total = 0
    with urllib.request.urlopen(url, timeout=45) as response, Path(path).open('xb') as stream:
        while block := response.read(1024**2):
            total += len(block)
            if total > maximum or shutil.disk_usage(ROOT).free < RESERVE:
                raise ValueError('Bounded source collection or original disk reserve exceeded')
            stream.write(block)
    return total


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])
            or OUT.exists() or DATA.exists() or shutil.disk_usage(ROOT).free < RESERVE + LIMIT):
        raise ValueError('Fresh bounded single-thread low-priority CPU collection required')
    OUT.mkdir(); DATA.mkdir()
    script_sha = digest(__file__)
    provenance = dict(
        dataset='nvidia/OpenCodeInstruct', revision=REVISION, source_file=FILE,
        exact_shard_sha256=SHA, exact_shard_bytes=BYTES, selected_complete_original_prefix=PREFIX,
        source_url=f'https://huggingface.co/datasets/nvidia/OpenCodeInstruct/tree/{REVISION}',
        paper_url='https://arxiv.org/html/2504.04030v1', publisher_declared_license='CC-BY-4.0',
        attribution='Wasi Uddin Ahmad et al., NVIDIA, OpenCodeInstruct (2025)',
        synthetic_teacher_data=True, published_response_generator='Qwen2.5-Coder-32B-Instruct',
        published_test_generator='Qwen2.5-Coder-32B-Instruct',
        published_generic_seed_instruction_generator='Qwen2.5-32B-Instruct',
        any_model_weights_downloaded=False, any_live_model_API_queried=False,
        human_preferences=False, own_model_on_policy=False, training_admitted=False,
        publisher_tests_and_pass_flags_are_not_independent_correctness=True,
        original_recipes_and_live_jobs_unchanged=True, new_neural_updates=0, goal_achieved=False)
    write(OUT / 'plan.json', dict(provenance, at=datetime.now(timezone.utc).isoformat(),
        producer_sha256=script_sha, minimum_free_bytes=RESERVE, maximum_new_bytes=LIMIT))
    try:
        tree_url=f'https://huggingface.co/api/datasets/nvidia/OpenCodeInstruct/tree/{REVISION}/data?limit=100'
        with urllib.request.urlopen(tree_url, timeout=30) as response:
            tree=json.load(response)
        write(DATA / 'publisher-file-tree.json', tree)
        selected=[r for r in tree if r['path']==FILE]
        if len(selected)!=1 or selected[0]['size']!=BYTES or selected[0]['lfs']['oid']!=SHA:
            raise ValueError('Pinned publisher shard metadata differs')
        base=f'https://huggingface.co/datasets/nvidia/OpenCodeInstruct/resolve/{REVISION}/'
        download(base+'README.md', DATA/'publisher-README.md', 2*1024**2)
        target=DATA/'train-00000-of-00050.parquet'
        download(base+FILE, target, BYTES)
        if target.stat().st_size!=BYTES or digest(target)!=SHA:
            raise ValueError('Whole downloaded public shard hash mismatch')
        parquet=pq.ParquetFile(target)
        rows=[]
        for batch in parquet.iter_batches(batch_size=128, use_threads=False):
            rows.extend(batch.to_pylist())
            if len(rows)>=PREFIX:break
        rows=rows[:PREFIX]
        if len(rows)!=PREFIX:raise ValueError('Complete fixed original prefix required')
        with (DATA/'first1024-original-entries.jsonl').open('x') as stream:
            for position, row in enumerate(rows):
                stream.write(json.dumps(dict(original_shard_position=position, original_entry=row),ensure_ascii=True)+'\n')
        if digest(__file__)!=script_sha:raise ValueError('Producer changed during collection')
        write(DATA/'provenance.json',provenance)
        artifact_hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(DATA.iterdir()) if p.is_file()}
        write(OUT/'complete.json',dict(provenance,complete=True,actual_shard_rows=parquet.metadata.num_rows,
            original_column_schema=str(parquet.schema_arrow),source_files_sha256={str(Path(__file__)):script_sha},
            artifact_sha256=artifact_hashes,original_entries_retained=PREFIX,
            at=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(dict(complete=True,original_entries=PREFIX,whole_shard_bytes=BYTES,training_admitted=False)))
    except BaseException as error:
        write(OUT/'failure.json',dict(error=repr(error),training_admitted=False,goal_achieved=False))
        raise


if __name__=='__main__':main()
