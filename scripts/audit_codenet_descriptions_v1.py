"""Measure original description export/sample coverage before dataset admission."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics

from tokenizers import Tokenizer

from slm_lab.codenet_description_v1 import parse_description
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'reports/codenet-description-format-v1'
    if folder.exists():
        raise ValueError('Preserve the earlier description audit')
    inspection = ROOT / 'reports/codenet-original-inspection-v1'
    plan = json.loads((inspection / 'plan.json').read_text())
    cache = Path(plan['metadata_cache'])
    receipt_path = ROOT / 'data/codenet-original-source-v1/source.json'
    receipt = json.loads(receipt_path.read_text())
    if plan['archive_sha256'] != receipt['sha256']:
        raise ValueError('Description cache refers to another original source')
    members = []
    with (cache / 'retained-members.jsonl').open() as stream:
        for line in stream:
            if not line.endswith('\n'):
                break  # The inspection journal may still be appending other sections.
            row = json.loads(line)
            if row['cache_relative_path'].startswith('problem_descriptions/'):
                members.append(row)
    members.sort(key=lambda r: r['member'])
    if len(members) != 3999 or len({r['member'] for r in members}) != 3999:
        raise ValueError('Expected all original description members from the completed prefix section')
    token_path = ROOT / 'checkpoints/instruction-generalization-probe-v3/candidate/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(token_path))
    folder.mkdir()
    counts = Counter()
    lengths = []
    records_path = folder / 'records.jsonl'
    spot_outputs = {'p00001': ['3776\n2848\n2840', '900\n900\n800'],
                    'p00092': ['5\n3'],
                    'p02167': ['1\n3\n6\n3\n6\n7', '431200\n629552\n431200\n629552\n275968\n101332\n528220'],
                    'p00098': ['16', '15']}
    with records_path.open('x') as stream:
        for member in members:
            path = cache / member['cache_relative_path']
            if path.stat().st_size != member['bytes'] or file_hash(path) != member['sha256']:
                raise ValueError('Retained original HTML differs')
            html = path.read_text(encoding='utf-8')
            parsed = parse_description(html)
            problem = path.stem
            if problem in spot_outputs and [s['output'] for s in parsed['samples']] != spot_outputs[problem]:
                raise ValueError('Manually inspected original sample outputs differ: ' + problem)
            tokens = len(tokenizer.encode(parsed['text']).ids)
            lengths.append(tokens)
            counts['original_descriptions'] += 1
            counts['recognized_sample_pairs'] += len(parsed['samples'])
            counts['recognized_samples_usable_files'] += parsed['recognized_samples_usable']
            counts['sample_issue_files'] += bool(parsed['sample_parse_issues'])
            counts['requires_media_review_files'] += parsed['requires_media_review']
            counts['empty_visible_text_files'] += not bool(parsed['text'])
            counts['plain_text_with_recognized_samples_files'] += bool(parsed['text']) and parsed['recognized_samples_usable'] and not parsed['requires_media_review']
            counts['no_pairs_despite_source_sample_cues_files'] += not parsed['samples'] and bool(re.search('sample|入力例|出力例', html, re.I))
            counts['problem_text_alone_exceeds_4096_tokens_files'] += tokens > 4096
            stream.write(json.dumps({'problem_id': problem, 'original_html_sha256': member['sha256'],
                'original_archive_member': member['member'], 'problem_only_tokens': tokens,
                **parsed}, ensure_ascii=False) + '\n')
    completed = inspection / 'inspection.json'
    crc_verified = False
    if completed.exists():
        complete = json.loads(completed.read_text())
        crc_verified = complete['archive_sha256'] == receipt['sha256'] and complete['whole_gzip_stream_drained_crc_checked']
    atomic_json(folder / 'summary.json', {
        'checked_at': datetime.now(timezone.utc).isoformat(), 'counts': dict(counts),
        'problem_only_token_counts': {'min': min(lengths), 'median': statistics.median(lengths), 'max': max(lengths)},
        'manually_inspected_output_spot_checks': list(spot_outputs),
        'source_archive_sha256': receipt['sha256'], 'source_receipt_sha256': file_hash(receipt_path),
        'description_member_subset_sha256': hashlib.sha256(json.dumps(members, sort_keys=True).encode()).hexdigest(),
        'tokenizer_sha256': file_hash(token_path), 'records_sha256': file_hash(records_path),
        'parser_sha256': file_hash(ROOT / 'src/slm_lab/codenet_description_v1.py'),
        'script_sha256': file_hash(Path(__file__)),
        'whole_archive_crc_verified_at_audit': bool(crc_verified),
        'dataset_admitted_to_training': False, 'model_training_started': False, 'goal_achieved': False,
        'scope': 'Description/sample format coverage only. Recognized samples are not all judge tests or correctness/efficiency proof. Text exports are not certified complete; media/markup issues remain for review. Token counts exclude prompts and program responses, so they cannot establish complete SFT context fit.'})
    print(json.dumps(dict(counts), indent=2))


if __name__ == '__main__':
    main()
