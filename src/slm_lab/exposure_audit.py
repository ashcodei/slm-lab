"""Read-only benchmark exposure scan across frozen packed text and SFT records."""
import codecs
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import re

import ahocorasick
import numpy as np
from tokenizers import Tokenizer

from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .evaluation_provenance import capture
from .model import ModelConfig
from .training import Encoder

ROOT = Path(__file__).resolve().parents[2]


def normalize(value):
    return ' '.join(value.casefold().split())


def byte_alphabet():
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    points = values.copy(); counter = 0
    for value in range(256):
        if value not in values:
            values.append(value); points.append(256 + counter); counter += 1
    return {chr(point): value for value, point in zip(values, points)}


class StreamMatcher:
    """Incremental UTF-8 and normalized-text carry; never cross EOS/BOS/PAD."""
    def __init__(self, tokenizer, patterns):
        raw = json.loads(tokenizer.to_str())
        if raw['model']['type'] != 'BPE' or raw['decoder']['type'] != 'ByteLevel':
            raise ValueError('Exposure audit requires the actual own byte-level BPE tokenizer')
        grouped = defaultdict(list)
        for key, value in patterns.items():
            text = normalize(value)
            if not text: raise ValueError('Empty benchmark text cannot define an exposure case')
            grouped[text].append(key)
        self.automaton = ahocorasick.Automaton()
        for text, keys in grouped.items(): self.automaton.add_word(text, (text, keys))
        self.automaton.make_automaton(); self.carry = max(map(len, grouped)) - 1
        alphabet = byte_alphabet(); added = {row['id']: row['content'].encode() for row in raw['added_tokens']}
        vocab = tokenizer.get_vocab(); self.bytes = [None] * (max(vocab.values()) + 1)
        for token, index in vocab.items():
            self.bytes[index] = added[index] if index in added else bytes(alphabet[char] for char in token)
        self.boundaries = {tokenizer.token_to_id(value) for value in ['<eos>', '<bos>', '<pad>']}
        self.boundaries.discard(None)
        self.decoder = codecs.getincrementaldecoder('utf-8')('replace')
        self.tail = ''; self.characters = 0; self.tokens = 0; self.document = 0; self.document_start = 0
        self.matches = {}

    def _text(self, value, shard):
        fresh = re.sub(r'\s+', ' ', value.casefold())
        if self.tail.endswith(' ') and fresh.startswith(' '): fresh = fresh[1:]
        text = self.tail + fresh; start_offset = self.characters - len(self.tail)
        for end, (pattern, keys) in self.automaton.iter(text):
            for key in keys:
                self.matches.setdefault(key, {'document_index': self.document,
                    'document_token_start': self.document_start, 'scan_token_end': self.tokens,
                    'match_normalized_start': start_offset + end - len(pattern) + 1,
                    'match_normalized_end': start_offset + end + 1, 'end_shard': shard,
                    'matched_text_sha256': hashlib.sha256(pattern.encode()).hexdigest()})
        self.characters += len(fresh)
        self.tail = text[-self.carry:] if self.carry else ''

    def feed(self, ids, shard='memory'):
        # Split only at actual control IDs. Incremental UTF-8 survives arbitrary
        # token/chunk/shard boundaries; normalized carry survives long whitespace.
        start = 0
        for index, token in enumerate(ids):
            if token not in self.boundaries: continue
            self._segment(ids[start:index], shard)
            self._text(self.decoder.decode(b'', final=True), shard)
            self.tokens += 1; self.document += 1; self.document_start = self.tokens
            self.decoder.reset(); self.tail = ''; self.characters = 0; start = index + 1
        self._segment(ids[start:], shard)

    def _segment(self, ids, shard):
        if not ids: return
        if any(token < 0 or token >= len(self.bytes) or self.bytes[token] is None for token in ids):
            raise ValueError('Packed source contains an ID outside the own tokenizer vocabulary')
        raw = b''.join(self.bytes[token] for token in ids)
        self.tokens += len(ids); self._text(self.decoder.decode(raw), shard)

    def finish(self):
        self._text(self.decoder.decode(b'', final=True), 'end-of-source')
        return self.matches


def scan_corpus(root, tokenizer, patterns, progress):
    root = Path(root); manifest_path = root / 'corpus.json'; manifest_sha = file_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest['tokenizer_sha256'] != file_hash(tokenizer): raise ValueError('Own training tokenizer changed')
    tok = Tokenizer.from_file(str(tokenizer)); matcher = StreamMatcher(tok, patterns)
    snapshots = {}; shards = manifest['splits']['train']['shards']
    for index, shard in enumerate(shards):
        path = root / shard['path']
        if file_hash(path) != shard['sha256']: raise ValueError('Actual packed training shard changed')
        values = np.memmap(path, dtype=manifest['dtype'], mode='r')
        if len(values) != shard['tokens']: raise ValueError('Packed source token count differs')
        snapshots[str(path.resolve())] = (shard['sha256'], path.stat().st_size, path.stat().st_mtime_ns)
        for start in range(0, len(values), 100000):
            matcher.feed(values[start:start + 100000].astype(np.int64).tolist(), shard['path'])
        if index % 25 == 0 or index + 1 == len(shards):
            progress({'corpus': str(root), 'shards_completed': index + 1, 'shards': len(shards),
                      'tokens_scanned': matcher.tokens, 'matched_benchmark_texts': len(matcher.matches)})
    matches = matcher.finish()
    if matcher.tokens != manifest['splits']['train']['tokens'] or file_hash(manifest_path) != manifest_sha:
        raise ValueError('Complete frozen packed-source inventory changed during scan')
    for name, (digest, size, modified) in snapshots.items():
        path = Path(name)
        if path.stat().st_size != size or path.stat().st_mtime_ns != modified or file_hash(path) != digest:
            raise ValueError('Scanned packed training bytes changed during audit')
    return {'corpus_manifest_sha256': manifest_sha, 'actual_training_tokens_scanned': matcher.tokens,
            'actual_training_shards_rehashed': len(shards), 'matches': matches}


def audit():
    reports = ROOT / 'reports/expanded-training-exposure-audit-v1'
    if reports.exists(): raise ValueError('Preserve existing exposure audit; inspect any live process before resuming')
    implementation_files = [ROOT / name for name in capture()['files_sha256']] + [Path(__file__)]
    runtime = capture(implementation_files)
    runtime['audit_packages'] = {name: version(name) for name in ['numpy', 'pyahocorasick']}
    source_files = {}; patterns = {}; benchmarks = {}
    for family, name, field in [
        ('knowledge', 'data/knowledge-eval-v1/questions.jsonl', 'question'),
        ('math', 'data/math-reasoning-v2/test.jsonl', 'question'),
        ('public_code', 'data/public-code-assessment-v1/questions.jsonl', 'specification')]:
        path = ROOT / name; source_files[str(path)] = file_hash(path); rows = list(jsonl(path))
        for row in rows:
            key = family + ':' + str(row['id'])
            if key in patterns: raise ValueError('Duplicate benchmark identity')
            patterns[key] = row[field]
        benchmarks[family] = {'questions': len(rows), 'source_sha256': source_files[str(path)], 'field': field}
    tokenizer = ROOT / 'data/tokenizer-corpus/tokenizer.json'; tokenizer_sha = file_hash(tokenizer)
    reports.mkdir()
    corpora = [
        ('past_broad_pretraining', ROOT / 'data/broad-mixture-v1'),
        ('past_foundation_warmup', ROOT / 'data/expanded-foundation-warmup-v1'),
        ('current_full_pretraining_plan', ROOT / 'data/extended-curriculum-v1/packed'),
        ('queued_library_code', ROOT / 'data/broad-code-continuation-v1/packed')]
    sft_files = sorted({Path(stage['data']) for stage in json.loads((ROOT / 'reports/extended-curriculum-v1/plan.json').read_text())['stages']
                        if stage['stage'] == 'sft'})
    sft_files += [ROOT / name for name in ['data/human-dialogue-v1/train.jsonl',
                  'data/exercism-code-v1/train.jsonl', 'data/textvqa-human-reading-v1/train.jsonl']]
    for path in sft_files: source_files[str(path)] = file_hash(path)
    implementation_sha = file_hash(__file__)
    atomic_json(reports / 'plan.json', {'started_at': datetime.now(timezone.utc).isoformat(),
               'benchmarks': benchmarks, 'source_files_sha256': source_files, 'tokenizer_sha256': tokenizer_sha,
               'implementation_sha256': implementation_sha, 'implementation_runtime': runtime,
               'packed_corpora': {name: str(path) for name, path in corpora},
               'scope': 'Exact casefold/whitespace normalized full benchmark question/specification exposure within EOS-separated documents. Current/queued data are scanned in full; not all tokens have been trained yet.',
               'goal_achieved': False})
    def progress(value):
        value.update(updated_at=datetime.now(timezone.utc).isoformat(), status='auditing expanded actual training inputs', goal_achieved=False)
        atomic_json(reports / 'status.json', value); print(json.dumps(value), flush=True)
    try:
        findings = {}
        for name, path in corpora:
            findings[name] = scan_corpus(path, tokenizer, patterns, progress)
            atomic_json(reports / (name + '.json'), findings[name])
        sft = {}
        tok = Tokenizer.from_file(str(tokenizer))
        config = ModelConfig.load(ROOT / 'configs/model.json')
        for path in sft_files:
            matcher = StreamMatcher(tok, patterns); count = 0
            encoder = Encoder(tok, config, path.parent, 448)
            # Use the real learner-visible sequence, including role formatting,
            # coordinate transforms and truncation. Do not scan gold/source metadata.
            for row in jsonl(path):
                ids, _ = encoder.sequence(row, 'response')
                matcher.feed(ids, path.name); count += 1
            sft[str(path.relative_to(ROOT))] = {'examples': count, 'matches': matcher.finish(),
                                               'source_sha256': source_files[str(path)],
                                               'truncated_training_sequences': encoder.truncated}
            progress({'sft_source': str(path), 'examples_scanned': count, 'matched_benchmark_texts': len(matcher.matches)})
        for path, digest in source_files.items():
            if file_hash(path) != digest: raise ValueError('Frozen benchmark/SFT source changed during audit')
        if file_hash(tokenizer) != tokenizer_sha or file_hash(__file__) != implementation_sha:
            raise ValueError('Loaded audit tokenizer/implementation changed')
        final_runtime = capture(implementation_files)
        final_runtime['audit_packages'] = {name: version(name) for name in ['numpy', 'pyahocorasick']}
        if final_runtime != runtime: raise ValueError('Audit encoder/decoder implementation or runtime changed')
        exposed = sorted({key for result in [*findings.values(), *sft.values()] for key in result['matches']})
        report = {'completed_at': datetime.now(timezone.utc).isoformat(), 'benchmarks': benchmarks,
                  'packed_findings': findings, 'sft_findings': sft, 'exposed_case_ids': exposed,
                  'source_files_sha256': source_files, 'tokenizer_sha256': tokenizer_sha,
                  'implementation_sha256': implementation_sha, 'implementation_runtime': runtime,
                  'benchmark_files_modified': False, 'goal_achieved': False,
                  'limitations': ['Exact full-question/specification exposure only; paraphrases, solutions and semantic overlap are not ruled out.',
                                  'Current/queued whole data plans include examples not yet trained; repeated token streams are counted separately.',
                                  'EOS/BOS/PAD are treated as boundaries, including any literal special token embedded in original source text.',
                                  'SFT scan uses actual encoder-visible sequences at448px; image pixels and source metadata are not searched.']}
        atomic_json(reports / 'audit.json', report)
        atomic_json(reports / 'status.json', {'status': 'expanded training exposure audit completed; broad generalization unverified',
                   'exposed_benchmark_cases': len(exposed), 'goal_achieved': False})
    except Exception as error:
        atomic_json(reports / 'status.json', {'status': 'expanded exposure audit failed; inspect without restarting live work',
                                            'error': str(error), 'goal_achieved': False}); raise


if __name__ == '__main__': audit()
