"""Reconstruct complete Go/Ruby interfaces before admitting source variants."""
import hashlib
import json
from pathlib import Path
import re

from tokenizers import Tokenizer

from .corpus import file_hash
from .diverse_images import jsonl
from .evaluation_provenance import capture
from .goruby_code_headers import HeaderParser
from .model import ModelConfig
from .training import Encoder

FENCE = re.compile(r'\A```(go|ruby)\n(.*)\n```\Z', re.S)
ORIGIN = 'Exact parsed original Go function/Ruby method header; source code response and metadata unchanged.'


def sha(text): return hashlib.sha256(text.encode()).hexdigest()


def reconstruct(original, parsed, encoder):
    if original['source'] != 'CodeSearchNet' or original['language'] not in ('Go', 'Ruby'):
        raise ValueError('Original documented Go/Ruby training code required')
    match = FENCE.fullmatch(original['response'])
    if match is None or not parsed['parsed'] or not parsed['needs_correction']:
        raise ValueError('Complete original code and a parsed header correction required')
    code = match.group(2); header = parsed['complete_header']
    if sha(code) != original['source_code_sha256'] or code.encode()[:parsed['header_end_byte']].decode() != header:
        raise ValueError('Original implementation or complete source header bytes differ')
    marker = 'Signature/context: ' + code.split('{', 1)[0][:500] + '\n\n'
    if marker not in original['prompt']: raise ValueError('Original documented signature field differs')
    prompt = original['prompt'].replace(marker, 'Signature/context: ' + header + '\n\n', 1)
    variant = {**original, 'id': sha(original['id'] + '\nfull-source-header-v1\n' + prompt + '\n' + original['response']),
        'prompt': prompt, 'split': 'train', 'parent_instruction_id': original['id'],
        'parent_prompt_sha256': sha(original['prompt']), 'parent_response_sha256': sha(original['response']),
        'task': 'complete_documented_code_header', 'complete_source_header': header,
        'header_end_byte': parsed['header_end_byte'], 'parser_node_type': parsed['node_type'],
        'original_header_text_cut': parsed['header_text_cut'],
        'original_method_body_in_signature': parsed['method_body_text_in_context'],
        'header_origin': ORIGIN, 'external_model_teacher_used': False}
    inputs, labels = encoder.sequence(variant, 'response')
    prompt_ids = encoder.tokens(encoder.prompt(variant)); response_ids = encoder.tokens(variant['response'])
    if (inputs != [encoder.bos] + prompt_ids + response_ids + [encoder.eos]
            or labels != [-100] * (len(prompt_ids) + 1) + response_ids + [encoder.eos]):
        raise ValueError('Complete corrected source interface/code would be clipped or mislabeled')
    return {**variant, 'actual_target_tokens': len(response_ids) + 1,
            'complete_input_sha256': sha(json.dumps(inputs, separators=(',', ':')))}, len(inputs)


def unchanged(actual, expected):
    if actual != expected: raise ValueError('Prepared Go/Ruby interface variant differs from original-source reconstruction')


def admit(root):
    root = Path(root).resolve(); data = root / 'data/complete-goruby-headers-v1'
    manifest_path = data / 'manifest.json'; manifest_sha = file_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    files = [root / name for name in manifest['preparation_provenance']['files_sha256']]
    if capture(files) != manifest['preparation_provenance']: raise ValueError('Header preparation source/runtime changed')
    for name, digest in manifest['files_sha256'].items():
        if file_hash(data / name) != digest: raise ValueError('Prepared complete Go/Ruby source rows changed')
    audit_path = root / 'reports/goruby-instruction-header-audit-v1.json'
    if file_hash(audit_path) != manifest['source_header_audit_sha256']: raise ValueError('Complete source grammar audit changed')
    audit = json.loads(audit_path.read_text())
    original_path = root / 'data/extended-curriculum-v1/code-instruction.train.jsonl'
    if file_hash(original_path) != manifest['source_data_sha256']: raise ValueError('Original code-instruction training inputs changed')
    config_path = root / 'configs/model.json'
    tokenizer_path = root / 'checkpoints/extended-curriculum-v1/foundation-warmup/tokenizer.json'
    if file_hash(config_path) != manifest['config_sha256'] or file_hash(tokenizer_path) != manifest['tokenizer_sha256']:
        raise ValueError('Complete header Encoder inputs differ')
    parser = HeaderParser(root)
    if parser.manifest_sha != manifest['grammar_toolchain_manifest_sha256']: raise ValueError('Pinned Go/Ruby grammars differ')
    files = list(dict.fromkeys(files + [Path(__file__)])); loaded = capture(files)
    rows = list(jsonl(data / 'train.jsonl')); parents = {row['parent_instruction_id']: row for row in rows}
    if len(rows) != 47006 or len(parents) != len(rows) or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Complete unique original Go/Ruby correction population required')
    originals = {}
    for row in jsonl(original_path):
        if row['id'] in parents:
            if row['id'] in originals: raise ValueError('Repeated original documented code parent')
            originals[row['id']] = row
    if set(originals) != set(parents): raise ValueError('Prepared header parent is not original training data')
    valid = root / 'data/code-search-expanded-v1/sft.valid.jsonl'
    original_manifest = json.loads((valid.parent / 'corpus.json').read_text())
    if file_hash(valid) != original_manifest['sft_files_sha256']['valid']: raise ValueError('Original held code source changed')
    groups = {row['group_id'] for row in rows}
    if any(row['group_id'] in groups for row in jsonl(valid)): raise ValueError('Corrected source parents overlap original held groups')
    encoder = Encoder(Tokenizer.from_file(str(tokenizer_path)), ModelConfig.load(config_path), data, 448)
    targets = maximum = 0; counts = {}
    for actual in rows:
        original = originals[actual['parent_instruction_id']]
        code = FENCE.fullmatch(original['response']).group(2)
        expected, length = reconstruct(original, parser.measure(original['language'], code), encoder)
        unchanged(actual, expected)
        targets += expected['actual_target_tokens']; maximum = max(maximum, length)
        counts[original['language']] = counts.get(original['language'], 0) + 1
    if (encoder.truncated or targets != manifest['targets_per_pass'] or maximum != manifest['maximum_complete_input_tokens']
            or counts != manifest['retained_by_language'] or manifest['train_examples'] != len(rows)
            or counts != {language: stats['needs_correction'] for language, stats in audit['counts'].items()}):
        raise ValueError('Complete original header coverage/target counts differ')
    if (file_hash(manifest_path) != manifest_sha or capture(files) != loaded
            or file_hash(original_path) != manifest['source_data_sha256']
            or file_hash(valid) != original_manifest['sft_files_sha256']['valid']
            or any(file_hash(data / name) != digest for name, digest in manifest['files_sha256'].items())):
        raise ValueError('Header admission source/runtime changed during reconstruction')
    return rows, {'passed': True, 'manifest_sha256': manifest_sha, 'examples': len(rows), 'by_language': counts,
        'actual_targets_per_source_pass': targets, 'maximum_complete_input_tokens': maximum,
        'all_original_targets_metadata_and_header_variants_reconstructed': True,
        'actual_complete_input_and_response_only_labels_verified': True, 'original_validation_group_overlap': 0,
        'evaluation_provenance': loaded, 'source_programs_executed': False, 'trained': False, 'goal_achieved': False,
        'scope': 'Original documented training-source header variants only; source algorithms and missing module dependencies remain unverified.'}
