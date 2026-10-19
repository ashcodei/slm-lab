"""Bind the files actually consumed by the current execution data owners.

Prerequisite proof JSON files retain their own immutable hashes. Their unrelated
historical training-state payloads are not inputs to this new code experiment.
Each active owner's explicit source map and every native pack file is retained.
"""
import json
from pathlib import Path

from .corpus import file_hash


def merge_bindings(destination, incoming):
    for name, digest in incoming.items():
        name = str(Path(name).resolve())
        if name in destination and destination[name] != digest:
            raise ValueError('Conflicting active runtime source bindings')
        destination[name] = digest


def input_bindings(root, corpus):
    root = Path(root).resolve(); result = {}
    # These source maps are explicitly checked by the active corpus/reward
    # constructors, rather than discovered by recursively crawling old reports.
    descriptors = [
        'data/apps-execution-rewards-v1/admission.json',
        'data/APPS-verified-generation-v1/admission.json',
        'data/documented-doctest-verified-v1/admission.json',
        'reports/bounded-code-probes-v1/admission.json',
        'reports/causal-code-rewards-v2/admission.json']
    for name in descriptors:
        path = root/name; metadata = json.loads(path.read_text())
        merge_bindings(result, metadata['source_files_sha256'])
        merge_bindings(result, {str(path):file_hash(path)})
    for pack in corpus.packs.values():
        for name in ['manifest.json','admission.json']:
            path = pack.folder/name; merge_bindings(result,{str(path):file_hash(path)})
        merge_bindings(result,{str(pack.folder/name):digest for name,digest in pack.manifest['files_sha256'].items()})
    # Directly read reward ledgers/readbacks remain explicit runtime inputs even
    # when their admission records store these as separate named SHA fields.
    names = [
        'data/apps-execution-rewards-v1/manifest.json', 'data/apps-execution-rewards-v1/catalog.jsonl',
        'data/apps-execution-rewards-v1/controls.jsonl', 'reports/apps-execution-rewards-v1/independent-reader.json',
        'data/documented-doctest-stage-v1/candidates.jsonl',
        'reports/documented-doctest-verified-v1/observations.jsonl',
        'reports/documented-doctest-verified-v1/independent-complete-source-and-runtime-readback-v1.json',
        'reports/causal-code-rewards-v2/observations.jsonl', 'reports/causal-code-rewards-v2/independent-reader.json',
        'reports/bounded-code-probes-v1/observations.jsonl', 'reports/bounded-code-probes-v1/independent-reader.json']
    for name in names:
        path=root/name; merge_bindings(result,{str(path):file_hash(path)})
    return result
