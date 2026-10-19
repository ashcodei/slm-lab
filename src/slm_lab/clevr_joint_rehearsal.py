"""Mix audited native scene descriptions with the exact earlier short-QA subset."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from tokenizers import Tokenizer

from .clevr_scene_evaluation import score, summarize
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .model import ModelConfig
from .training import Encoder

ROOT = Path(__file__).resolve().parents[2]


def question_subset(rows, count=10000, seed=42):
    if not 0 < count <= len(rows):
        raise ValueError('QA budget must fit the original source')
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [rows[index] for index in order[:count]]


def compare_scene_reports(before, after):
    for key in ['evaluation_provenance', 'data_sha256', 'manifest_sha256', 'tokenizer_sha256',
                'image_size', 'inference_precision', 'inference_device_type',
                'max_new_tokens', 'max_examples', 'protocol']:
        if before[key] != after[key]:
            raise ValueError('Unmatched scene assessment: ' + key)

    def cases(report):
        rows = report['results']
        if not rows or len({row['id'] for row in rows}) != len(rows):
            raise ValueError('Empty or repeated scene cases')
        for row in rows:
            for control in ['proper', 'blank', 'wrong']:
                if row[control + '_scores'] != score(row[control + '_generated'], row['expected']):
                    raise ValueError('Scene scores differ from actual free generation')
        if summarize(rows) != report['overall']:
            raise ValueError('Scene aggregate differs from case evidence')
        return [(row['id'], row['group_id'], row['expected'], row['prompt_sha256'],
                 row['input_pixels_sha256'], row['wrong_pixels_sha256'],
                 row['wrong_image_id'], row['target_tokens']) for row in rows]

    if cases(before) != cases(after):
        raise ValueError('Unmatched scene cases/pixels/controls')
    lost = {}
    for metric in ['valid_format', 'full_scene_exact', 'object_count_exact', 'shape_counts_exact']:
        lost[metric] = [left['id'] for left, right in zip(before['results'], after['results'])
                        if left['proper_scores'][metric] and not right['proper_scores'][metric]]
    return {'before': before['overall'], 'after': after['overall'],
            'previously_correct_cases_lost': lost,
            'passed': not any(lost.values()), 'goal_achieved': False,
            'scope': 'Reused rendered development scenes; no real-photo capability or promotion.'}


def prepare(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve existing rehearsal inputs; inspect before any resume')
    source = ROOT / 'data/clevr-reasoning-v1'
    dense = ROOT / 'data/clevr-dense-scenes-v1'
    manifests = {name: json.loads((directory / 'manifest.json').read_text())
                 for name, directory in [('questions', source), ('descriptions', dense)]}
    initial_hashes = {}
    for name, directory in [('questions', source), ('descriptions', dense)]:
        initial_hashes[str(directory / 'manifest.json')] = file_hash(directory / 'manifest.json')
        for filename in ['train.jsonl', 'valid.jsonl']:
            path = directory / filename
            digest = file_hash(path)
            if digest != manifests[name]['files_sha256'][filename]:
                raise ValueError('Original source split changed')
            initial_hashes[str(path)] = digest
    audit_path = ROOT / 'reports/clevr-original-input-audit-v1.json'
    audit = json.loads(audit_path.read_text())
    if audit.get('passed') is not True or audit['manifest_sha256'] != initial_hashes[str(source / 'manifest.json')]:
        raise ValueError('Original scene/photo audit does not match')
    questions = question_subset(list(jsonl(source / 'train.jsonl')))
    descriptions = list(jsonl(dense / 'train.jsonl'))
    protected = {row['group_id'] for directory in [source, dense] for row in jsonl(directory / 'valid.jsonl')}
    tokenizer = ROOT / 'data/tokenizer-corpus/tokenizer.json'
    tokenizer_sha = file_hash(tokenizer)
    if any(manifest['tokenizer_sha256'] != tokenizer_sha for manifest in manifests.values()):
        raise ValueError('Own tokenizer differs from source preparation')
    encoder = Encoder(Tokenizer.from_file(str(tokenizer)), ModelConfig.load(ROOT / 'configs/model.json'), output, 448)
    rows = []
    image_hashes = {}
    targets = {'questions': 0, 'descriptions': 0}
    for family, directory, examples in [('questions', source, questions), ('descriptions', dense, descriptions)]:
        for original in examples:
            row = dict(original)
            image = (directory / row['image']).resolve()
            if not image.is_relative_to(source.resolve()) or row['group_id'] in protected:
                raise ValueError('Training photo path/group escapes the audited training source')
            if image not in image_hashes:
                image_hashes[image] = file_hash(image)
            if image_hashes[image] != row['stored_image_sha256']:
                raise ValueError('Audited training image changed or source records conflict')
            row['image'] = str(image)
            row['rehearsal_family'] = family
            ids, labels = encoder.sequence(row, 'response')
            if encoder.truncated:
                raise ValueError('Complete rehearsal example does not fit the own context')
            row['rehearsal_context_tokens'] = len(ids)
            row['rehearsal_target_tokens'] = sum(value != -100 for value in labels[1:])
            targets[family] += row['rehearsal_target_tokens']
            rows.append(row)
    if len(rows) != 14000 or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Rehearsal count/unique source record IDs differ')
    if targets != {'questions': 24707, 'descriptions': 311386}:
        raise ValueError('Exact earlier QA exposures or full dense-description targets differ')
    for path, digest in initial_hashes.items():
        if file_hash(path) != digest:
            raise ValueError('Original input changed during preparation')
    output.mkdir(parents=True)
    path = output / 'train.jsonl'
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    manifest = {'prepared_at': datetime.now(timezone.utc).isoformat(), 'implementation_sha256': file_hash(__file__),
                'original_input_audit_sha256': file_hash(audit_path), 'source_files_sha256': initial_hashes,
                'tokenizer_sha256': tokenizer_sha, 'files_sha256': {'train.jsonl': file_hash(path)},
                'train_examples': len(rows), 'training_image_groups': len({row['group_id'] for row in rows}),
                'family_examples': {'questions': len(questions), 'descriptions': len(descriptions)},
                'family_target_tokens': targets, 'target_tokens': sum(targets.values()),
                'qa_source_selection_seed': 42,
                'qa_exposure_ids_sha256': hashlib.sha256('\n'.join(row['id'] for row in questions).encode()).hexdigest(),
                'validation_group_overlap': 0, 'steps_one_pass_batch2_accumulation4': len(rows) // 8,
                'external_model_teacher_used': False, 'goal_achieved': False,
                'limitations': ['Extra 500 updates and 311386 description targets versus QA-only; unmatched budget.',
                                'Existing loss weights target tokens: descriptions account for about 92.65% of supervised tokens.',
                                'Original rendered scenes; neither general real-image ability nor independent final testing.']}
    atomic_json(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    print(json.dumps(prepare(ROOT / 'data/clevr-joint-rehearsal-v1'), indent=2))
