"""Parse all frozen Go/Ruby source targets without executing source programs."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation_provenance import capture
from slm_lab.goruby_code_headers import HeaderParser

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r'\A```(go|ruby)\n(.*)\n```\Z', re.S)


def main():
    output = ROOT / 'reports/goruby-instruction-header-audit-v1.json'
    if output.exists(): raise ValueError('Preserve an existing completed grammar audit')
    plan_path = ROOT / 'reports/extended-curriculum-v1/plan.json'
    stage = next(row for row in json.loads(plan_path.read_text())['stages'] if row['name'] == 'multilingual-code-instruction')
    source = Path(stage['data']); digest = file_hash(source)
    if digest != stage['data_sha256']: raise ValueError('Frozen code instructions changed')
    parser = HeaderParser(ROOT)
    probe = json.loads((ROOT / 'reports/goruby-header-parser-probes-v1.json').read_text())
    if probe['passed'] is not True or probe['toolchain_manifest_sha256'] != parser.manifest_sha:
        raise ValueError('Actual grammar header probes must pass first')
    files = [Path(__file__), ROOT / 'src/slm_lab/goruby_code_headers.py', parser.manifest_path,
        ROOT / 'scripts/verify_goruby_header_parser.py']
    files += [Path(parser.metadata['toolchain_root']) / name for name in parser.metadata['installed_files_sha256']]
    runtime = capture(files); counts = defaultdict(Counter); examples = defaultdict(list); total = 0
    started = time.monotonic()
    for index, row in enumerate(jsonl(source), 1):
        total = index
        if row['source'] != 'CodeSearchNet' or row.get('language') not in ('Go', 'Ruby'): continue
        language = row['language']; stats = counts[language]; stats['source_rows'] += 1
        match = FENCE.fullmatch(row['response'])
        if match is None: raise ValueError('Original complete fenced Go/Ruby source required')
        code = match.group(2); measured = parser.measure(language, code)
        if not measured['parsed']:
            stats['unsupported_source_or_context'] += 1
            key = language + '_unsupported'
            if len(examples[key]) < 10:
                examples[key].append({'id': row['id'], 'source_url': row['source_url'], **measured})
            continue
        if 'Signature/context: ' + measured['original_signature'] + '\n\n' not in row['prompt']:
            raise ValueError('Actual prompt differs from the original signature transform')
        stats['parsed_outer_interfaces'] += 1
        for key in ('header_text_cut', 'parameter_text_cut', 'source_after_header_in_context',
                    'method_body_text_in_context', 'needs_correction'):
            stats[key] += int(measured[key])
            sample = language + '_' + key
            if measured[key] and len(examples[sample]) < 10:
                examples[sample].append({'id': row['id'], 'source_url': row['source_url'],
                    'original_signature': measured['original_signature'], 'complete_header': measured['complete_header'],
                    'node_type': measured['node_type']})
        if stats['source_rows'] % 10000 == 0:
            atomic_json(ROOT / 'reports/goruby-instruction-header-audit-v1.progress.json',
                {'updated_at': datetime.now(timezone.utc).isoformat(), 'all_rows_scanned': index,
                 'counts': {k: dict(v) for k, v in counts.items()}, 'goal_achieved': False})
    if file_hash(source) != digest or capture(files) != runtime:
        raise ValueError('Actually audited source or grammar runtime changed')
    report = {'completed_at': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic() - started,
        'all_frozen_rows_scanned': total, 'source_data_sha256': digest, 'plan_sha256': file_hash(plan_path),
        'counts': {k: dict(v) for k, v in counts.items()}, 'examples': dict(examples),
        'grammar_toolchain_manifest_sha256': parser.manifest_sha, 'evaluation_provenance': runtime,
        'source_programs_executed': False, 'training_data_changed': False, 'goal_achieved': False,
        'primary_sources': ['https://tree-sitter.github.io/py-tree-sitter/',
            'https://github.com/tree-sitter/tree-sitter-go/tree/v0.25.0',
            'https://github.com/tree-sitter/tree-sitter-ruby/tree/v0.23.1'],
        'limitations': ['Unsupported grammar/context shapes are explicit, not proof of invalid original source.',
            'Source text after a Ruby interface can include comments, implementation statements or an end marker.',
            'Parsed headers do not prove documentation sufficiency, dependency validity, algorithmic correctness or model capability.',
            'Noncontiguous Ruby heredoc default arguments require further handling and are not silently rewritten.']}
    atomic_json(output, report)
    print(json.dumps({'report': str(output), 'counts': report['counts']}), flush=True)


if __name__ == '__main__': main()
