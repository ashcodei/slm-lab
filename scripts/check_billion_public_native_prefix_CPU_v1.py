"""Snapshot and independently rebuild 1024 real documents; never admit a prefix."""
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/public-billion-native-prefix-CPU-v1'
SOURCE = ROOT/'data/public-billion-causal-candidate-v2'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('Fresh GPU-hidden real public-source prefix check required')
    import numpy as np
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    from slm_lab.held_text_overlap_v1 import HeldTextOverlap
    from slm_lab.public_billion_readback_v1 import Rebuilder
    from slm_lab.public_causal_windows_v1 import CausalPublicPack, windows
    from read_and_pack_billion_public_candidate_CPU_v1 import QUOTAS
    tests = ROOT/'reports/public-causal-windows-and-billion-readback-v1-final-tests.log'
    if '7 passed' not in tests.read_text():raise ValueError('Actual native/mask/role/source controls required')
    source = json.loads((ROOT/'reports/public-billion-causal-candidate-CPU-v2/source-catalogs.json').read_text())['code'][0]
    original_path = ROOT/'data/public-pretraining-source-preflight-v1/code/original.parquet'
    if file_hash(original_path) != source['sha256'] or original_path.stat().st_size != source['size']:
        raise ValueError('Actual complete original source Parquet must retain its pinned publisher bytes')
    tokenizer_path = ROOT/'data/tokenizer-corpus/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    registry_path = ROOT/'reports/CodeContests-native-held-text-v2/unchanged-held-text-registry.json'
    registry = json.loads(registry_path.read_text())
    if len(registry) != 32953:raise ValueError('Every original declared held stem required')
    overlap = HeldTextOverlap(registry)
    OUT.mkdir()
    folder = SOURCE/'whole-native-documents'
    offsets = dict(train=0, valid=0)
    records = []
    # Snapshot immutable written prefixes; never alter the live collector files.
    with ExitStack() as stack:
        record_input = stack.enter_context(gzip.open(folder/'whole-original-documents.jsonl.gz', 'rt'))
        raw_input = {r:stack.enter_context((folder/(r+'.tokens.u16')).open('rb')) for r in offsets}
        raw_output = {r:stack.enter_context((OUT/(r+'.tokens.u16')).open('xb')) for r in offsets}
        docs = stack.enter_context((OUT/'original-native-documents.jsonl').open('x'))
        for _ in range(1024):
            record = json.loads(next(record_input))
            if record['original_source_file_sha256'] != source['sha256']:
                raise ValueError('Every declared snapshot document must belong to the pinned first public source')
            role = record['role'];length = record['native_token_length']
            if record['native_token_offset'] != offsets[role]:raise ValueError('Exact written native prefix required')
            raw = raw_input[role].read(length*2)
            if len(raw) != length*2:raise ValueError('The actual complete native prefix must already be durably written')
            raw_output[role].write(raw);offsets[role] += length
            docs.write(json.dumps(record, ensure_ascii=False)+'\n');records.append(record)
    prefix = records[-1]['original_parquet_row_ordinal']+1
    ledger = []
    with gzip.open(SOURCE/'observed-original-prefix-dispositions.jsonl.gz', 'rt') as stream, (OUT/'original-prefix-dispositions.jsonl').open('x') as output:
        for p in range(prefix):
            row = json.loads(next(stream))
            if row['source_file'] != 1 or row['original_row_ordinal'] != p:raise ValueError('Every original source prefix position required')
            ledger.append(row);output.write(json.dumps(row)+'\n')
    with ExitStack() as stack:
        streams = {r:stack.enter_context((OUT/(r+'.tokens.u16')).open('rb')) for r in offsets}
        documents = iter(json.loads(line) for line in (OUT/'original-native-documents.jsonl').open())
        rebuild = Rebuilder(QUOTAS, tokenizer, overlap, documents, streams)
        ordinal = 0
        for batch in pq.ParquetFile(original_path).iter_batches(batch_size=128, use_threads=False):
            for row in batch.to_pylist():
                rebuild.observe(row, source, ordinal, 1, ledger[ordinal]);ordinal += 1
                if ordinal == prefix:break
            if ordinal == prefix:break
        if (ordinal != prefix or next(documents, None) is not None
                or any(stream.read(1) for stream in streams.values()) or rebuild.offsets != offsets):
            raise ValueError('Every retained source row, complete original native sequence and byte tail must rebuild')
    roles = {}
    for role in offsets:
        starts = np.asarray(rebuild.starts[role], dtype='<u8')
        index = windows(offsets[role], starts, 4096)
        np.save(OUT/(role+'.windows.npy'), index, allow_pickle=False)
        np.save(OUT/(role+'.document-starts.npy'), starts, allow_pickle=False)
        roles[role] = dict(documents=len(starts), windows=len(index), native_context_tokens=offsets[role],
                           target_tokens=int(index['targets'].astype(np.uint64).sum()))
    files = [p for p in OUT.iterdir() if p.is_file()]
    manifest_path = OUT/'manifest.json'
    atomic_json(manifest_path, dict(format='own-public-causal-windows-v1', context_limit=4096, roles=roles,
        native_files={r:str(OUT/(r+'.tokens.u16')) for r in offsets},
        source_files_sha256={str(p):file_hash(p) for p in files}, optimization_admitted=False,
        prefix_only_not_billion_target_admission=True, model_updates=0, goal_achieved=False))
    window_proofs = {}
    for role in offsets:
        pack = CausalPublicPack(OUT, role, 'audit')
        counts = Counter()
        for p in range(len(pack)):
            ids, labels = pack.sequence(p)
            counts.update(targets=sum(v != -100 for v in labels[1:]), EOS=sum(v == 2 for v in labels[1:]), windows=1)
        positions = sorted({0, min(1, len(pack)-1), len(pack)-1})
        batch = pack.batch(positions)
        if int((batch['labels'][:, 1:] != -100).sum()) != sum(int(pack.index[p]['targets']) for p in positions):
            raise ValueError('Actual native collated tensor batch targets differ')
        if counts['targets'] != roles[role]['target_tokens'] or counts['EOS'] != roles[role]['documents']:
            raise ValueError('Every original text and EOS target must occur once in actual bounded windows')
        window_proofs[role] = dict(counts)
    languages = dict(Counter(r['language'] for r in records))
    if len(languages) != 8 or not languages.get('Go') or not languages.get('Rust'):
        raise ValueError('The actual declared 1024-document prefix must exercise all eight programming languages')
    atomic_json(OUT/'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), complete=True,
        actual_original_documents=1024, actual_original_prefix_rows_rebuilt=prefix, actual_code_languages=languages,
        actual_original_publisher_GO_rows=sum(r['original_publisher_language']=='GO' for r in records),
        actual_independently_rebuilt_prefix=rebuild.summary(), actual_complete_window_proofs=window_proofs,
        actual_original_publisher_fields_native_bytes_masks_EOS_roles_rights_and_source_dispositions_rebuilt=True,
        source_files_sha256={str(Path(__file__)):file_hash(Path(__file__)), str(original_path):file_hash(original_path),
            str(tokenizer_path):file_hash(tokenizer_path), str(registry_path):file_hash(registry_path),
            str(manifest_path):file_hash(manifest_path), str(tests):file_hash(tests)},
        this_is_a1024document_prefix_not_complete_billion_target_readback=True,
        optimization_admitted=False, neural_training_updates=0, useful_model_gain_not_measured=True, goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT/'complete.json'), actual_original_documents=1024,
        actual_original_rows=prefix, actual_all_eight_code_languages=languages,
        actual_train_targets=roles['train']['target_tokens'], optimization_admitted=False)), flush=True)


if __name__ == '__main__':main()
