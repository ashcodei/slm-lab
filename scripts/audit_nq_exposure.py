"""Check original NQ development questions against owned past/planned training text."""
from datetime import datetime, timezone
import gzip
from importlib.metadata import version
import json
from pathlib import Path

from tokenizers import Tokenizer

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation_provenance import capture
from slm_lab.exposure_audit import StreamMatcher, scan_corpus
from slm_lab.model import ModelConfig
from slm_lab.training import Encoder

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'reports/nq-development-exposure-v1'
    if output.exists(): raise ValueError('Preserve existing NQ exposure audit; inspect its actor before resuming')
    data = ROOT / 'data/nq-human-questions-v1'; manifest_path = data / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()); dev = data / 'valid.jsonl'
    if file_hash(dev) != manifest['files_sha256']['valid.jsonl']: raise ValueError('Prepared original development questions changed')
    patterns = {row['id']: row['source_original_question']['question'] for row in jsonl(dev)}
    if len(patterns) != 3610: raise ValueError('All original NQ-open development questions required')
    tokenizer_path = ROOT / 'data/tokenizer-corpus/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path)); config = ModelConfig.load(ROOT / 'configs/model.json')
    corpora = [('past_broad_pretraining', ROOT / 'data/broad-mixture-v1'),
               ('past_foundation_warmup', ROOT / 'data/expanded-foundation-warmup-v1'),
               ('current_full_pretraining_plan', ROOT / 'data/extended-curriculum-v1/packed'),
               ('queued_library_code', ROOT / 'data/broad-code-continuation-v1/packed')]
    records = {}
    for stage in json.loads((ROOT / 'reports/extended-curriculum-v1/plan.json').read_text())['stages']:
        if stage['stage'] in ['sft', 'contrastive']: records[Path(stage['data'])] = stage['stage']
    for name in ['instruction', 'images', 'joint', 'fashion-only']:
        records[ROOT / 'data/alignment-curriculum-v1' / (name + '.train.jsonl')] = 'sft'
    records[ROOT / 'data/alignment-curriculum-v1/contrastive.train.jsonl'] = 'contrastive'
    for name in ['human-dialogue-v1', 'exercism-code-v1', 'textvqa-human-reading-v1',
                 'expanded-everyday-photos-v1', 'expanded-scene-words-v1', 'clevr-reasoning-v1',
                 'visual-pretraining-v1', 'nq-human-questions-v1']:
        records[ROOT / 'data' / name / 'train.jsonl'] = 'sft'
    records[ROOT / 'data/own-ocr-human-questions-v1/train.jsonl.gz'] = 'sft'
    preferences = ROOT / 'data/human-preference-curriculum-v1/preferences.train.jsonl'
    ppo = ROOT / 'data/human-preference-curriculum-v1/ppo.train.jsonl'
    source_files = {str(path): file_hash(path) for path in [manifest_path, dev, tokenizer_path, ROOT / 'configs/model.json', *records, preferences, ppo]}
    files = [ROOT / name for name in capture()['files_sha256']] + [Path(__file__), ROOT / 'src/slm_lab/exposure_audit.py']
    runtime = capture(files); packages = {name: version(name) for name in ['numpy', 'pyahocorasick']}
    output.mkdir()
    atomic_json(output / 'plan.json', {'started_at': datetime.now(timezone.utc).isoformat(),
        'benchmark_questions': len(patterns), 'source_files_sha256': source_files,
        'packed_corpora': {name: str(path) for name, path in corpora},
        'record_sources': {str(path): mode for path, mode in records.items()},
        'implementation_runtime': runtime, 'audit_packages': packages, 'goal_achieved': False,
        'scope': 'Casefold/whitespace normalized full original development-question exposure in complete past/current/queued sources, including the new prepared training set; not all planned data has been trained.'})

    def progress(**fields):
        value = {'status': 'scanning NQ development exposure in actual frozen training inputs',
                 'updated_at': datetime.now(timezone.utc).isoformat(), 'goal_achieved': False, **fields}
        atomic_json(output / 'status.json', value); print(json.dumps(value), flush=True)

    def scan_rows(path, mode, choice=None):
        matcher = StreamMatcher(tokenizer, patterns); encoder = Encoder(tokenizer, config, path.parent, 448); count = 0
        if path.suffix == '.gz':
            def rows():
                with gzip.open(path, 'rt') as stream:
                    for line in stream: yield json.loads(line)
        else:
            rows = lambda: jsonl(path)
        for row in rows():
            if mode == 'ppo':
                body = encoder.tokens(encoder.prompt(row))[-(config.max_text_length - 256 - 1):]
                ids = [encoder.bos] + body + [encoder.eos]
            elif mode == 'contrastive':
                ids, _ = encoder.sequence({**row, 'text': row['response']}, None)
            else:
                ids, _ = encoder.sequence({**row, 'response': row[choice]} if choice else row, 'response')
            matcher.feed(ids, str(path)); count += 1
            if count % 50000 == 0: progress(source=str(path), mode=mode, choice=choice, rows_scanned=count, matched_questions=len(matcher.matches))
        return {'source_sha256': source_files[str(path)], 'mode': mode, 'choice': choice,
                'examples': count, 'truncated_sequences': encoder.truncated, 'matches': matcher.finish()}

    try:
        packed = {}; encoded = {}
        for name, path in corpora:
            packed[name] = scan_corpus(path, tokenizer_path, patterns, lambda fields: progress(**fields))
            atomic_json(output / (name + '.json'), packed[name])
        for index, (path, mode) in enumerate(records.items()):
            result = scan_rows(path, mode); encoded[str(path.relative_to(ROOT))] = result
            atomic_json(output / ('encoded-source-' + str(index) + '.json'), result)
            progress(source=str(path), rows_scanned=result['examples'], matched_questions=len(result['matches']))
        for choice in ['chosen', 'rejected']:
            encoded['human_feedback_' + choice] = scan_rows(preferences, 'preference', choice)
        encoded['PPO_prompts'] = scan_rows(ppo, 'ppo')
        for path, digest in source_files.items():
            if file_hash(path) != digest: raise ValueError('NQ benchmark/frozen training input changed during exposure scan')
        if capture(files) != runtime or {name: version(name) for name in packages} != packages:
            raise ValueError('NQ exposure scanner/encoder/runtime changed')
        exposed = sorted({key for result in [*packed.values(), *encoded.values()] for key in result['matches']})
        report = {'completed_at': datetime.now(timezone.utc).isoformat(), 'completed': True, 'benchmark_questions': 3610,
                  'benchmark_data_sha256': source_files[str(dev)], 'source_files_sha256': source_files,
                  'packed_findings': packed, 'encoded_findings': encoded, 'exposed_case_ids': exposed,
                  'implementation_runtime': runtime, 'audit_packages': packages, 'goal_achieved': False,
                  'limitations': ['Exact full-question text only; answer/solution/concept/paraphrase/article overlap is not ruled out.',
                                  'Past/current/queued whole sources are conservative potential exposure inventories; overlapping streams and repeats are counted separately.',
                                  'PPO sampled continuations, image pixels and source metadata are not scanned.',
                                  'Casefold/whitespace matching does not promise Unicode compatibility-normalized equivalence.',
                                  'Exposed original development cases remain in the benchmark and must be identified separately; this audit cannot make the whole split independently unseen.']}
        atomic_json(output / 'audit.json', report)
        progress(status_summary='NQ development exposure audit completed; broad generalization unverified', exposed_questions=len(exposed))
    except Exception as error:
        progress(status_summary='NQ exposure audit failed; inspect without restarting live work', error=str(error)); raise


if __name__ == '__main__': main()
