"""Check committed original photo pairs without finalizing or admitting the live collection."""
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.model import ModelConfig
from slm_lab.training import Encoder
from slm_lab.vqa_pair_admission import admit, reconstruct_question, unchanged, verified_photo
from slm_lab.vqa_pair_evaluation import validate_pairs
from slm_lab.vqa_pairs import canonical_question

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    data = ROOT / 'data/vqa-real-pairs-v1'
    # Snapshot complete newline-committed journal records; the collector may append.
    journal = (data / 'images.collected.jsonl').read_bytes()
    if not journal.endswith(b'\n'): journal = journal.rsplit(b'\n', 1)[0] + b'\n'
    photos = [json.loads(line) for line in journal.splitlines()]
    records = {(row['source_subset'], row['coco_image_id']): row for row in photos}
    if len(records) != len(photos): raise ValueError('Repeated committed source photo')
    encoder = Encoder(Tokenizer.from_file(str(ROOT / 'data/tokenizer-corpus/tokenizer.json')),
                      ModelConfig.load(ROOT / 'configs/model.json'), data, 448)
    verified = set(); splits = {}
    for split in ('train', 'valid'):
        candidates = list(jsonl(ROOT / 'data/vqa-source-cache-v1' / ('candidate-pairs.' + split + '.jsonl')))
        complete = [pair for pair in candidates if all((pair['source_subset'], pair[side]['image']['id']) in records
                                                       for side in ('first', 'second'))]
        rows = {}; pairs = []
        for pair in complete:
            ids = []
            for side in ('first', 'second'):
                key = (pair['source_subset'], pair[side]['image']['id']); record = records[key]
                if key not in verified: verified_photo(data, record); verified.add(key)
                row = reconstruct_question(pair, side, record, encoder); ids.append(row['id'])
                if row['id'] in rows:
                    unchanged({k: v for k, v in rows[row['id']].items() if k != 'pair_ids'},
                              {k: v for k, v in row.items() if k != 'pair_ids'})
                    rows[row['id']]['pair_ids'].append(pair['id'])
                else: rows[row['id']] = row
            pairs.append({'id': pair['id'], 'question_ids': ids, 'canonical_question': canonical_question(pair),
                          'source_subset': pair['source_subset']})
        validate_pairs(pairs, list(rows.values()))
        splits[split] = {'complete_candidate_pairs': len(pairs), 'reconstructed_original_questions': len(rows),
                         'complete_response_targets': sum(row['target_tokens'] for row in rows.values())}
    final_absent = not (data / 'manifest.json').exists()
    if final_absent:
        try: admit(ROOT)
        except ValueError as error:
            if 'Finalized original real-photo pair manifest required' not in str(error): raise
        else: raise ValueError('Unfinalized photo collection incorrectly admitted')
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'preview_checks_passed': True,
              'snapshot_committed_photos': len(photos), 'actually_verified_used_photos': len(verified), 'splits': splits,
              'admission_source_sha256': file_hash(ROOT / 'src/slm_lab/vqa_pair_admission.py'),
              'final_manifest_absent': final_absent, 'unfinalized_admission_rejection_verified': final_absent,
              'training_admitted': False, 'trained': False, 'goal_achieved': False,
              'scope': 'Read-only append-journal preview of original question, creator declaration, stored pixels and complete Encoder masks. Final collector source/split/perceptual protections and final manifest remain required; no training or model capability claim.'}
    atomic_json(ROOT / 'reports/vqa-pair-admission-preview-v1.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
