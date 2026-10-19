"""Reconstruct every prepared prompt/answer pair before admitting context SFT."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

from .corpus import file_hash
from .diverse_images import jsonl
from .evaluation_provenance import capture
from .model import ModelConfig
from .own_ocr_training_data import variants
from .textvqa_admission import admit as admit_original
from .training import Encoder


def unchanged_variant(actual, expected):
    if actual != expected: raise ValueError('Prepared human answer/image/question/predicted-context variant differs from reconstruction')


def admit(root):
    root = Path(root).resolve(); data = root / 'data/own-ocr-human-questions-v1'
    path = data / 'manifest.json'; manifest_sha = file_hash(path); manifest = json.loads(path.read_text())
    files = [root / name for name in manifest['source_provenance']['files_sha256']]
    if capture(files) != manifest['source_provenance']: raise ValueError('Own-context preparation implementation/runtime changed')
    original_train, original_receipt = admit_original(root)
    original_data = root / 'data/textvqa-human-reading-v1'
    if file_hash(original_data / 'manifest.json') != manifest['original_human_question_manifest_sha256']: raise ValueError('Original human-question manifest changed')
    for source, digest in {**manifest['original_question_rows_sha256'], **manifest['own_component_sha256']}.items():
        if file_hash(source) != digest: raise ValueError('Original human rows/own prediction model changed')
    for name, digest in manifest['files_sha256'].items():
        if file_hash(data / name) != digest: raise ValueError('Prepared own-context compressed rows/predictions changed')
    tokenizer_path = root / 'data/tokenizer-corpus/tokenizer.json'
    if file_hash(tokenizer_path) != manifest['tokenizer_sha256']: raise ValueError('Own context tokenizer changed')
    tokenizer = Tokenizer.from_file(str(tokenizer_path)); config = ModelConfig.load(root / 'configs/model.json')
    encoder = Encoder(tokenizer, config, data, 448); rows = {'train': original_train, 'valid': list(jsonl(original_data / 'valid.jsonl'))}
    contexts = {}; report_splits = {}
    for split in ['train', 'valid']:
        predictions = {}
        with gzip.open(data / (split + '.predictions.jsonl.gz'), 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if row['group_id'] in predictions or row['reference_answers_supplied_at_prediction'] is not False:
                    raise ValueError('Repeated or reference-bearing original own photo predictions')
                predictions[row['group_id']] = row
        expected = {}
        for original in rows[split]:
            prediction = predictions[original['group_id']]
            if prediction['image'] != original['image'] or prediction['stored_image_sha256'] != original['stored_image_sha256']:
                raise ValueError('Own predicted context refers to a different actual human-question photo')
            for variant in variants(original, prediction['predicted_words'], tokenizer, config):
                _, target = encoder.sequence(variant, 'response')
                if encoder.truncated: raise ValueError('Reconstructed human question/answer truncated')
                variant['target_tokens'] = sum(value != -100 for value in target[1:])
                if variant['id'] in expected: raise ValueError('Repeated reconstructed original question variant')
                expected[variant['id']] = variant
        if set(predictions) != {row['group_id'] for row in rows[split]}: raise ValueError('Incomplete/extra original own photo prediction population')
        decoded = hashlib.sha256(); seen = set(); retained = []; stats = Counter()
        with gzip.open(data / (split + '.jsonl.gz'), 'rb') as stream:
            for line in stream:
                decoded.update(line); row = json.loads(line)
                if row['id'] in seen or row['id'] not in expected: raise ValueError('Repeated/unknown original human variant')
                unchanged_variant(row, expected[row['id']]); seen.add(row['id'])
                stats['variants'] += 1; stats['target_token_exposures'] += row['target_tokens']
                if row['own_predicted_context_supplied']:
                    retained.append(row); stats['own_context_variants'] += 1
        if seen != set(expected) or decoded.hexdigest() != manifest['decoded_JSONL_sha256'][split + '.jsonl']:
            raise ValueError('Prepared human variant coverage/decoded bytes changed')
        if (len(retained) != len(rows[split]) or len(seen) != manifest['splits'][split]['examples']
                or stats['target_token_exposures'] != manifest['splits'][split]['target_token_exposures_per_pass']):
            raise ValueError('Original human variant/exposure counts differ')
        contexts[split] = retained; report_splits[split] = dict(stats)
    if {row['group_id'] for row in contexts['train']} & {row['group_id'] for row in contexts['valid']}: raise ValueError('Own-context human photo groups overlap training/validation')
    if file_hash(path) != manifest_sha or capture(files) != manifest['source_provenance']: raise ValueError('Own-context source changed during reconstruction')
    receipt = {'passed': True, 'manifest_sha256': manifest_sha, 'original_human_input_admission': original_receipt,
               'splits': report_splits, 'all_question_answer_prompt_image_variants_reconstructed': True,
               'validation_rows_not_returned_for_training': True, 'goal_achieved': False,
               'scope': 'Exact prompts rebuilt from recorded own predictions, original human questions/answers and fixed context budgets; no independent neural prediction reimplementation or capability claim.'}
    return contexts['train'], receipt
