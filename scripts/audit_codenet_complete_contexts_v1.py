"""Measure complete original problem/program context fit with actual response masks."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from slm_lab.codenet_complete_context_v1 import CompleteCodeEncoder, CompleteContextDoesNotFit, problem_program_row
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'reports/codenet-complete-contexts-v1'
    if folder.exists():
        raise ValueError('Preserve the earlier full-context assessment')
    folder.mkdir()
    def status(message, **fields):
        atomic_json(folder / 'status.json', {'status': message,
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'split_assignment_completed': False, 'model_training_started': False,
            'goal_achieved': False, **fields})
    try:
        body_path = ROOT / 'reports/codenet-candidate-bodies-v1/bodies.json'
        group_path = ROOT / 'reports/codenet-problem-groups-v2/groups.json'
        meta_manifest = ROOT / 'data/codenet-candidate-selection-v1/manifest.json'
        desc_summary = ROOT / 'reports/codenet-description-format-v1/summary.json'
        body = json.loads(body_path.read_text())
        grouping = json.loads(group_path.read_text())
        meta = json.loads(meta_manifest.read_text())
        desc = json.loads(desc_summary.read_text())
        cache = Path(body['candidate_cache'])
        source = cache / 'source-bodies.bin'
        index = cache / 'source-index.jsonl'
        metadata_path = meta_manifest.with_name('selected-metadata.jsonl')
        descriptions_path = desc_summary.with_name('records.jsonl')
        token_path = ROOT / 'checkpoints/instruction-generalization-probe-v3/candidate/tokenizer.json'
        config_path = ROOT / 'configs/model.json'
        expected = {source: body['source_body_store_sha256'], index: body['source_index_sha256'],
            metadata_path: meta['selected_metadata_sha256'], descriptions_path: desc['records_sha256'],
            token_path: desc['tokenizer_sha256']}
        if not body['whole_gzip_stream_drained_crc_checked'] or grouping['all_candidate_body_joins_verified'] != body['source_bodies']:
            raise ValueError('Exact completed body and grouping assessments required')
        if body['selected_metadata_sha256'] != meta['selected_metadata_sha256']:
            raise ValueError('Original selected cohort differs')
        for path, digest in expected.items():
            if file_hash(path) != digest:
                raise ValueError('Prepared complete-context source bytes differ')
        paths = [Path(__file__), ROOT / 'src/slm_lab/codenet_complete_context_v1.py',
            ROOT / 'src/slm_lab/training.py', ROOT / 'src/slm_lab/model.py',
            config_path, body_path, group_path, meta_manifest, desc_summary]
        bound = {**expected, **{p: file_hash(p) for p in paths}}
        states = {p: (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in bound}
        def guard(full=False):
            for p, digest in bound.items():
                now = p.stat()
                if (now.st_ino, now.st_size, now.st_mtime_ns) != states[p] or (full and file_hash(p) != digest):
                    raise ValueError('Bound complete-context input/runtime changed')
        descriptions = {r['problem_id']: r for r in (json.loads(line) for line in descriptions_path.open())}
        metadata = {r['archive_member']: r['metadata'] for r in (json.loads(line) for line in metadata_path.open())}
        config = ModelConfig.load(config_path)
        tokenizer = Tokenizer.from_file(str(token_path))
        encoder = CompleteCodeEncoder(tokenizer, config, ROOT)
        special = {row['id'] for row in json.loads(token_path.read_text())['added_tokens'] if row.get('special')}
        atomic_json(folder / 'plan.json', {'recorded_at': datetime.now(timezone.utc).isoformat(),
            'bound_files_sha256': {str(p): d for p, d in bound.items()},
            'candidates': len(metadata), 'complete_text_window': config.max_text_length,
            'programme_targets_are_original_unmodified_source': True,
            'split_assignment_completed': False, 'model_training_started': False, 'goal_achieved': False})
        counts = Counter()
        by_language = defaultdict(Counter)
        target_total = context_total = longer_targets = largest = 0
        seen = set()
        outliers = {member for values in grouping['clone_coverage'].values()
                    for member in values['selected_publisher_size_outliers']}
        record_path = folder / 'records.jsonl'
        with source.open('rb') as programs, index.open() as entries, record_path.open('x') as output:
            for line in entries:
                entry = json.loads(line)
                member = entry['archive_member']
                if member in seen or member not in metadata or programs.tell() != entry['body_offset']:
                    raise ValueError('Complete source-body population/offset join differs')
                seen.add(member)
                raw = programs.read(entry['body_bytes'])
                if len(raw) != entry['body_bytes'] or hashlib.sha256(raw).hexdigest() != entry['body_sha256']:
                    raise ValueError('Actual original program bytes differ')
                m = metadata[member]
                problem = descriptions.get(m['problem_id'])
                record = {'archive_member': member, 'problem_id': m['problem_id'], 'language': m['language'],
                    'proposed_problem_group': grouping['problem_to_proposed_group'][m['problem_id']],
                    'publisher_size_outlier': member in outliers, 'split': None}
                reason = None
                if m['language'] == 'Text': reason = 'output_only_text'
                elif problem is None: reason = 'original_description_missing'
                elif not problem['text'].strip(): reason = 'original_visible_problem_empty'
                elif problem['requires_media_review']: reason = 'original_media_review_needed'
                elif any('Unclosed' in issue['reason'] for issue in problem['sample_parse_issues']): reason = 'original_markup_review_needed'
                elif not entry['utf8_valid']: reason = 'source_encoding_review_needed'
                elif entry['contains_nul'] or not raw.strip(): reason = 'non_program_or_empty_source_review_needed'
                if reason is None:
                    row = problem_program_row(problem, m, raw.decode('utf-8'))
                    try:
                        ids, labels = encoder.sequence(row, 'response')
                    except CompleteContextDoesNotFit as error:
                        reason = 'complete_pair_exceeds_text_window'
                        record.update(prompt_tokens=error.prompt_tokens, response_tokens=error.response_tokens,
                                      complete_pair_tokens=error.prompt_tokens + error.response_tokens + 2)
                    else:
                        if len(ids) != len(labels) or ids[0] != encoder.bos or ids[-1] != encoder.eos:
                            raise ValueError('Actual full target framing differs')
                        start = next(i for i, value in enumerate(labels) if value != -100)
                        if labels[:start] != [-100] * start or labels[start:] != ids[start:]:
                            raise ValueError('Actual complete response mask differs')
                        targets = len(ids) - start
                        record.update(context_tokens=len(ids), target_tokens=targets,
                            complete_input_ids_sha256=hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest(),
                            target_longer_than_legacy_half_window=targets - 1 > config.max_text_length // 2 - 1)
                        if any(token in special for token in ids[1:-1]):
                            reason = 'literal_reserved_token_review_needed'
                        else:
                            reason = 'complete_original_pair_fits'
                            context_total += len(ids)
                            target_total += targets
                            longer_targets += record['target_longer_than_legacy_half_window']
                            largest = max(largest, len(ids))
                record['context_qualification'] = reason
                counts[reason] += 1
                by_language[m['language']][reason] += 1
                output.write(json.dumps(record) + '\n')
                if len(seen) % 1000 == 0:
                    guard()
                    output.flush()
                    status('measuring original complete problem/program contexts; no training',
                           checked_candidates=len(seen), expected_candidates=len(metadata), actual_counts=dict(counts))
        if seen != set(metadata) or encoder.truncated:
            raise ValueError('Candidate coverage incomplete or any target clipped')
        guard(full=True)
        atomic_json(folder / 'summary.json', {'completed_at': datetime.now(timezone.utc).isoformat(),
            'candidates_checked': len(seen), 'context_qualifications': dict(counts),
            'by_original_language': {name: dict(v) for name, v in by_language.items()},
            'complete_fit_context_tokens': context_total, 'complete_fit_target_tokens': target_total,
            'complete_fit_targets_longer_than_legacy_half_window': longer_targets,
            'largest_complete_fit_context': largest, 'encoder_truncated_sequences': encoder.truncated,
            'records_sha256': file_hash(record_path), 'plan_sha256': file_hash(folder / 'plan.json'),
            'split_assignment_completed': False, 'dataset_admitted_to_training': False,
            'model_training_started': False, 'goal_achieved': False,
            'scope': 'Actual complete text/mask eligibility only. Original source status is historical judge evidence. Semantic completeness, supported execution, efficiency and previous-corpus contamination are not certified. Pending media/long-context rows remain retained; no model capability gain claimed.'})
        status('complete original context assessment finished; final dataset admission pending', actual_counts=dict(counts))
        print(json.dumps(dict(counts), indent=2))
    except Exception as error:
        status('complete-context assessment stopped; requires inspection', error=str(error))
        raise


if __name__ == '__main__':
    main()
