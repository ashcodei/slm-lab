"""Rebuild corrected parameter prompts from original training source before SFT."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

from tokenizers import Tokenizer

from .corpus import file_hash
from .diverse_images import jsonl
from .evaluation_provenance import capture
from .model import ModelConfig
from .training import Encoder
from .javascript_recovered_toolchain_v2 import validate_toolchain

FENCE = re.compile(r'\A```javascript\n(.*)\n```\Z', re.S)
ORIGIN = 'Exact parsed source header before the outer function/method body; original code response and source metadata unchanged.'


def sha(text): return hashlib.sha256(text.encode()).hexdigest()


def reconstruct(original, parsed, encoder):
    if original['source'] != 'CodeSearchNet' or original['language'] != 'JavaScript':
        raise ValueError('Corrected JavaScript parameters require original documented training code')
    match = FENCE.fullmatch(original['response'])
    if match is None or not parsed['parsed'] or not parsed['parameter_text_cut']:
        raise ValueError('Full original code and a parsed incomplete parameter header required')
    code = match.group(1); header = parsed['original_full_header']
    if sha(code) != original['source_code_sha256'] or not code.startswith(header):
        raise ValueError('Original JavaScript implementation or parsed header changed')
    marker = 'Signature/context: ' + code.split('{', 1)[0][:500] + '\n\n'
    if marker not in original['prompt']: raise ValueError('Original incomplete signature differs')
    prompt = original['prompt'].replace(marker, 'Signature/context: ' + header + '\n\n', 1)
    variant = {**original, 'id': sha(original['id'] + '\n' + prompt + '\n' + original['response']), 'prompt': prompt,
        'parent_instruction_id': original['id'], 'parent_prompt_sha256': sha(original['prompt']),
        'parent_response_sha256': sha(original['response']), 'task': 'complete_documented_code_parameters',
        'complete_source_header': header, 'parser_node_type': parsed['node_type'], 'parser_context': parsed['context'],
        'parameter_header_origin': ORIGIN, 'external_model_teacher_used': False}
    inputs, labels = encoder.sequence(variant, 'response')
    prompt_ids = encoder.tokens(encoder.prompt(variant)); response_ids = encoder.tokens(variant['response'])
    if (inputs != [encoder.bos] + prompt_ids + response_ids + [encoder.eos]
            or labels != [-100] * (len(prompt_ids) + 1) + response_ids + [encoder.eos]):
        raise ValueError('Corrected full parameter prompt/code target would be clipped or mislabeled')
    return {**variant, 'split': 'train', 'complete_input_sha256': sha(json.dumps(inputs, separators=(',', ':'))),
            'actual_target_tokens': len(response_ids) + 1}, len(inputs)


def unchanged(actual, expected):
    if actual != expected: raise ValueError('Corrected parameter row differs from original-source reconstruction')


def admit(root):
    root = Path(root).resolve(); data = root / 'data/complete-js-parameters-v1'
    manifest_path = data / 'manifest.json'; manifest_sha = file_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads((data / 'candidate-receipt.json').read_text())
    proof = json.loads((root / 'reports/complete-js-parameter-variant-reconstruction-v1.json').read_text())
    if proof['manifest_sha256'] != manifest_sha or proof['all982_original_parent_targets_and_metadata_independently_verified'] is not True:
        raise ValueError('Completed independent original parameter reconstruction required')
    for name, digest in manifest['files_sha256'].items():
        if file_hash(data / name) != digest: raise ValueError('Prepared complete-parameter rows changed')
    if file_hash(data / 'candidate-receipt.json') != manifest['candidate_receipt_sha256']:
        raise ValueError('Original parameter preparation receipt changed')
    toolchain_proof = validate_toolchain(root, receipt['toolchain_manifest_sha256'], receipt['parser_version'])
    for name, field in [('scripts/prepare_js_parameter_variants.cjs', 'implementation_sha256'),
                         ('scripts/js_instruction_header.cjs', 'helper_sha256')]:
        if file_hash(root / name) != receipt[field]: raise ValueError('Parameter parser/preparation identity changed')
    audit_path = root / 'reports/javascript-instruction-interface-audit-v1.json'
    if file_hash(audit_path) != manifest['source_interface_audit_sha256']:
        raise ValueError('Complete original JavaScript interface audit changed')
    audit = json.loads(audit_path.read_text())
    if file_hash(root / 'scripts/audit_js_instruction_interfaces.cjs') != audit['implementation_sha256']:
        raise ValueError('Audited original signature implementation changed')
    original_path = root / 'data/extended-curriculum-v1/code-instruction.train.jsonl'
    if file_hash(original_path) != manifest['source_data_sha256'] or receipt['source_data_sha256'] != manifest['source_data_sha256']:
        raise ValueError('Original frozen code training inputs changed')
    config_path = root / 'configs/model.json'
    tokenizer_path = root / 'checkpoints/extended-curriculum-v1/foundation-warmup/tokenizer.json'
    if file_hash(config_path) != manifest['config_sha256'] or file_hash(tokenizer_path) != manifest['tokenizer_sha256']:
        raise ValueError('Actual complete-parameter Encoder configuration changed')
    files = [root / name for name in capture()['files_sha256']]
    files += [Path(__file__), Path(__file__).with_name('javascript_recovered_toolchain_v2.py'),
              config_path, tokenizer_path, root / 'scripts/js_instruction_header.cjs',
              root / 'scripts/prepare_js_parameter_variants.cjs', root / 'scripts/audit_js_instruction_interfaces.cjs',
              root / '.tools/javascript-toolchain-v1.json']
    runtime = capture(files)
    rows = list(jsonl(data / 'train.jsonl')); wanted = {row['parent_instruction_id']: row for row in rows}
    if len(rows) != 982 or len(wanted) != 982 or len({row['id'] for row in rows}) != 982:
        raise ValueError('All982 unique original corrected parameter variants required')
    originals = {}
    for row in jsonl(original_path):
        if row['id'] in wanted:
            if row['id'] in originals: raise ValueError('Repeated original parameter parent')
            originals[row['id']] = row
    if set(originals) != set(wanted): raise ValueError('Corrected parameter parent is not in frozen training')
    original_validation = root / 'data/code-search-expanded-v1/sft.valid.jsonl'
    original_manifest = json.loads((original_validation.parent / 'corpus.json').read_text())
    if file_hash(original_validation) != original_manifest['sft_files_sha256']['valid']:
        raise ValueError('Original code validation records changed')
    groups = {row['group_id'] for row in rows}
    if any(row['group_id'] in groups for row in jsonl(original_validation)):
        raise ValueError('Corrected code parents overlap original held source-file groups')
    tools = json.loads((root / '.tools/javascript-toolchain-v1.json').read_text())
    command = "const fs=require('node:fs'); const h=require('./scripts/js_instruction_header.cjs'); const codes=JSON.parse(fs.readFileSync(0,'utf8')); process.stdout.write(JSON.stringify(codes.map(h.measure)));"
    codes = [FENCE.fullmatch(originals[row['parent_instruction_id']]['response']).group(1) for row in rows]
    process = subprocess.run([tools['node_path'], '-e', command], input=json.dumps(codes), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=root, check=True, timeout=60,
        env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'})
    parsed = json.loads(process.stdout)
    if len(parsed) != len(rows): raise ValueError('Incomplete original source-header parsing')
    encoder = Encoder(Tokenizer.from_file(str(tokenizer_path)), ModelConfig.load(config_path), data, 448)
    targets = maximum = 0
    for actual, header in zip(rows, parsed):
        expected, length = reconstruct(originals[actual['parent_instruction_id']], header, encoder)
        unchanged(actual, expected); targets += expected['actual_target_tokens']; maximum = max(maximum, length)
    if (encoder.truncated or targets != manifest['targets_per_pass'] or maximum != manifest['maximum_complete_input_tokens']
            or manifest['train_examples'] != len(rows)):
        raise ValueError('Complete corrected parameter token exposure/length counts differ')
    if (file_hash(manifest_path) != manifest_sha or capture(files) != runtime
            or validate_toolchain(root, receipt['toolchain_manifest_sha256'], receipt['parser_version']) != toolchain_proof
            or file_hash(original_path) != manifest['source_data_sha256']
            or file_hash(original_validation) != original_manifest['sft_files_sha256']['valid']
            or any(file_hash(data / name) != digest for name, digest in manifest['files_sha256'].items())):
        raise ValueError('Parameter admission evidence/runtime changed during reconstruction')
    admission = {'passed': True, 'manifest_sha256': manifest_sha, 'examples': len(rows),
        'actual_targets_per_pass': targets, 'maximum_complete_input_tokens': maximum,
        'all_original_targets_metadata_and_corrected_headers_reconstructed': True,
        'actual_complete_input_and_response_only_masks_verified': True, 'original_validation_group_overlap': 0,
        'source_data_sha256': manifest['source_data_sha256'], 'source_programs_executed': False,
        'evaluation_provenance': runtime, 'verified_toolchain_recovery': toolchain_proof,
        'trained': False, 'goal_achieved': False,
        'scope': 'Only original training-source variants with exact parsed parameter headers; source implementations remain unexecuted and semantically unverified.'}
    return rows, admission
