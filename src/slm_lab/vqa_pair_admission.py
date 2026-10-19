"""Rebuild original paired photo questions, creator declarations and model inputs."""
import gzip
import json
from pathlib import Path

from PIL import Image
from tokenizers import Tokenizer

from .corpus import file_hash
from .diverse_images import jsonl, pixel_hash
from .evaluation_provenance import capture
from .model import ModelConfig
from .training import Encoder
from .vqa_attribution import attribution
from .vqa_pair_evaluation import human_agreement, validate_pairs
from .vqa_pairs import canonical_question, known_photo_aliases, source_identity


def verified_photo(data, record):
    data = Path(data).resolve()
    for name, digest in [('image', 'stored_image_sha256'), ('attribution_page', 'attribution_page_sha256')]:
        path = (data / record[name]).resolve()
        if not path.is_relative_to(data) or file_hash(path) != record[digest]:
            raise ValueError('Original paired photo/attribution path or bytes changed')
    with gzip.open(data / record['attribution_page'], 'rb') as stream:
        raw = stream.read(2_000_001)
    credit = attribution(raw, record['source_flickr_photo_id'], record['source_url'])
    if any(record[key] != value for key, value in credit.items()):
        raise ValueError('Original creator/license declaration differs from stored attribution')
    if record['group_id'] != 'flickr:' + record['source_flickr_photo_id']:
        raise ValueError('Canonical source-photo identity differs')
    with Image.open(data / record['image']) as image:
        for size, key in [(224, 'model_pixel_sha256'), (448, 'model_pixel_sha256_448')]:
            if pixel_hash(image, size) != record[key]: raise ValueError('Actual original model-input photo pixels changed')


def reconstruct_question(pair, side, record, encoder):
    sample = pair[side]; question = canonical_question(pair)
    source = sample['image']; original = sample['question']; annotation = sample['annotation']
    if (record['source_subset'] != pair['source_subset'] or record['coco_image_id'] != source['id']
            or record['original_COCO_image_metadata'] != source or original['image_id'] != source['id']
            or annotation['image_id'] != source['id'] or annotation['question_id'] != original['question_id']
            or known_photo_aliases(source) != {record['source_flickr_photo_id']} or source['license'] != 4):
        raise ValueError('Original paired question/answer/photo/source identities differ')
    answers = annotation['answers']; human_agreement(sample['answer'], answers)
    votes = sum(answer['answer'] == sample['answer'] for answer in answers)
    if votes < 6 or votes != sample['exact_human_votes']:
        raise ValueError('Original paired answer lacks its exact declared human agreement')
    identity = 'vqa2:' + pair['source_subset'] + ':question:' + str(original['question_id'])
    row = {**record, 'id': identity, 'pair_ids': [pair['id']], 'prompt': question + '\nAnswer briefly.',
        'response': sample['answer'], 'task': 'visual_qa', 'domain': 'everyday_photos',
        'original_question': original, 'original_human_annotation': annotation,
        'exact_human_agreement': votes, 'annotation_license': 'CC-BY-4.0',
        'annotation_source': 'VQA Consortium original v2.0', 'annotation_source_url': 'https://visualqa.org/download.html',
        'external_model_teacher_used': False}
    inputs, labels = encoder.sequence(row, 'response')
    prompt = encoder.tokens(encoder.prompt(row)); response = encoder.tokens(row['response'])
    if (inputs != [encoder.bos] + prompt + response + [encoder.eos]
            or labels != [-100] * (len(prompt) + 1) + response + [encoder.eos]):
        raise ValueError('Complete original photo question/human answer would be clipped or mislabeled')
    row['context_tokens'] = len(inputs); row['target_tokens'] = len(response) + 1
    return row


def unchanged(actual, expected):
    if actual != expected: raise ValueError('Prepared original photo question differs from complete source reconstruction')


def admit(root):
    root = Path(root).resolve(); data = root / 'data/vqa-real-pairs-v1'; path = data / 'manifest.json'
    if not path.exists(): raise ValueError('Finalized original real-photo pair manifest required before training admission')
    digest = file_hash(path); manifest = json.loads(path.read_text())
    configuration_path = data / 'configuration.json'
    if json.loads(configuration_path.read_text()) != manifest['configuration']:
        raise ValueError('Final original photo-pair configuration differs')
    required = {'train.jsonl', 'valid.jsonl', 'pairs.train.jsonl', 'pairs.valid.jsonl', 'images.collected.jsonl'}
    if set(manifest['files_sha256']) != required:
        raise ValueError('Final original paired manifest lacks its complete file inventory')
    if source_identity() != manifest['configuration']['implementation_sources_sha256']:
        raise ValueError('Original photo-pair preparation source/runtime changed')
    for filename, expected in manifest['sources_sha256'].items():
        if file_hash(filename) != expected: raise ValueError('Original publisher/protected-photo source changed')
    for name, expected in manifest['files_sha256'].items():
        if file_hash(data / name) != expected: raise ValueError('Final original paired rows or photo journal changed')
    tokenizer_path = root / 'data/tokenizer-corpus/tokenizer.json'
    if file_hash(tokenizer_path) != manifest['tokenizer_sha256']: raise ValueError('Original photo-pair own tokenizer differs')
    files = [root / name for name in capture()['files_sha256']] + [Path(__file__),
        root / 'src/slm_lab/vqa_pairs.py', root / 'src/slm_lab/vqa_attribution.py',
        root / 'src/slm_lab/vqa_pair_evaluation.py', root / 'src/slm_lab/clevr_evaluation.py']
    runtime = capture(files)
    encoder = Encoder(Tokenizer.from_file(str(tokenizer_path)), ModelConfig.load(root / 'configs/model.json'), data, 448)
    images = {}
    for record in jsonl(data / 'images.collected.jsonl'):
        key = (record['source_subset'], record['coco_image_id'])
        if key in images: raise ValueError('Repeated original committed photo record')
        images[key] = record
    all_rows = {}; reports = {}; groups = {}; aliases = {}; pixels = {}; verified = set()
    for split in ('train', 'valid'):
        candidate_path = root / 'data/vqa-source-cache-v1' / ('candidate-pairs.' + split + '.jsonl')
        if manifest['sources_sha256'].get(str(candidate_path)) != file_hash(candidate_path):
            raise ValueError('Original candidate question source is not bound by the final manifest')
        candidate_rows = list(jsonl(candidate_path))
        candidates = {pair['id']: pair for pair in candidate_rows}
        if len(candidates) != len(candidate_rows): raise ValueError('Repeated original candidate pair identity')
        prepared = list(jsonl(data / (split + '.jsonl'))); pairs = list(jsonl(data / ('pairs.' + split + '.jsonl')))
        validate_pairs(pairs, prepared); expected_rows = {}; used_images = set()
        for pair_row in pairs:
            original = candidates.get(pair_row['id'])
            if original is None: raise ValueError('Original finalized visual pair is absent from publisher candidate source')
            qids = []
            for side in ('first', 'second'):
                sample = original[side]; key = (original['source_subset'], sample['image']['id'])
                if key not in images: raise ValueError('Original paired photo is not committed')
                record = images[key]
                if key not in verified: verified_photo(data, record); verified.add(key)
                row = reconstruct_question(original, side, record, encoder); qids.append(row['id']); used_images.add(key)
                if row['id'] in expected_rows:
                    previous = expected_rows[row['id']]
                    unchanged({k: v for k, v in previous.items() if k != 'pair_ids'},
                              {k: v for k, v in row.items() if k != 'pair_ids'})
                    previous['pair_ids'].append(original['id'])
                else: expected_rows[row['id']] = row
            if pair_row != {'id': original['id'], 'question_ids': qids,
                    'canonical_question': canonical_question(original), 'source_subset': original['source_subset']}:
                raise ValueError('Final original paired-question membership differs')
        if len(prepared) != len(expected_rows) or {row['id'] for row in prepared} != set(expected_rows):
            raise ValueError('Original photo-question population is incomplete or repeated')
        for row in prepared: unchanged(row, expected_rows[row['id']])
        stats = manifest['splits'][split]
        if not prepared or len(prepared) != stats['questions'] or len(pairs) != stats['pairs'] or len(used_images) != stats['images']:
            raise ValueError('Original paired source/photo counts differ')
        groups[split] = {row['group_id'] for row in prepared}
        aliases[split] = set().union(*(known_photo_aliases(row) for row in prepared))
        pixels[split] = {row[field] for row in prepared for field in ('model_pixel_sha256', 'model_pixel_sha256_448')}
        reports[split] = {'questions': len(prepared), 'pairs': len(pairs), 'source_photos': len(used_images),
                         'actual_target_tokens_per_source_pass': sum(row['target_tokens'] for row in prepared)}
        all_rows[split] = prepared
    if groups['train'] & groups['valid'] or aliases['train'] & aliases['valid'] or pixels['train'] & pixels['valid']:
        raise ValueError('Original paired training/validation photo identities or exact pixels overlap')
    if (encoder.truncated or file_hash(path) != digest or capture(files) != runtime
            or json.loads(configuration_path.read_text()) != manifest['configuration']
            or any(file_hash(filename) != expected for filename, expected in manifest['sources_sha256'].items())
            or any(file_hash(data / name) != expected for name, expected in manifest['files_sha256'].items())
            or file_hash(tokenizer_path) != manifest['tokenizer_sha256']):
        raise ValueError('Original paired source/Encoder/evidence changed during reconstruction')
    return [{**row, 'image': str((data / row['image']).resolve()), 'split': 'train'} for row in all_rows['train']], {
        'passed': True, 'manifest_sha256': digest, 'splits': reports,
        'all_original_question_answer_photo_pair_rows_reconstructed': True,
        'actual_complete_input_and_response_only_masks_verified': True,
        'original_creator_declarations_and_stored_pixels_verified': True,
        'source_photo_alias_and_exact224_448px_train_validation_overlap': 0,
        'validation_rows_not_returned_for_training': True, 'evaluation_provenance': runtime,
        'trained': False, 'goal_achieved': False,
        'scope': 'Finalized original paired human/photo inputs; declarations, perceptual screening and development controls do not certify ownership, semantic independence or learned grounding.'}
