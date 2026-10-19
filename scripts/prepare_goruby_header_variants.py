"""Create full-header instruction variants with unchanged original code labels."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from tokenizers import Tokenizer

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation_provenance import capture
from slm_lab.goruby_code_headers import HeaderParser
from slm_lab.model import ModelConfig
from slm_lab.training import Encoder

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r'\A```(go|ruby)\n(.*)\n```\Z', re.S)


def main():
    directory = ROOT / 'data/complete-goruby-headers-v1'
    if directory.exists(): raise ValueError('Preserve existing header variants; inspect before changing')
    audit_path = ROOT / 'reports/goruby-instruction-header-audit-v1.json'
    audit = json.loads(audit_path.read_text()); source = ROOT / 'data/extended-curriculum-v1/code-instruction.train.jsonl'
    digest = file_hash(source)
    if digest != audit['source_data_sha256']: raise ValueError('Audited original source inputs changed')
    parser = HeaderParser(ROOT)
    if parser.manifest_sha != audit['grammar_toolchain_manifest_sha256']: raise ValueError('Audited grammars changed')
    audit_files = [ROOT / name for name in audit['evaluation_provenance']['files_sha256']]
    if capture(audit_files) != audit['evaluation_provenance']: raise ValueError('Audited header implementation/runtime changed')
    tokenizer_path = ROOT / 'checkpoints/extended-curriculum-v1/foundation-warmup/tokenizer.json'
    config_path = ROOT / 'configs/model.json'
    encoder = Encoder(Tokenizer.from_file(str(tokenizer_path)), ModelConfig.load(config_path), directory, 448)
    files = [ROOT / name for name in capture()['files_sha256']] + audit_files + [Path(__file__), tokenizer_path, config_path]
    files = list(dict.fromkeys(files)); runtime = capture(files)
    directory.mkdir(); output = directory / 'train.jsonl'
    candidate_counts = Counter(); retained_counts = Counter(); exclusions = []; targets = maximum = 0; seen = set()
    with output.open('x') as destination:
        for original in jsonl(source):
            if original['source'] != 'CodeSearchNet' or original.get('language') not in ('Go', 'Ruby'): continue
            match = FENCE.fullmatch(original['response'])
            if match is None: raise ValueError('Expected complete original Go/Ruby code fence')
            code = match.group(2); header = parser.measure(original['language'], code)
            if not header['parsed'] or not header['needs_correction']: continue
            language = original['language']; candidate_counts[language] += 1
            if hashlib.sha256(code.encode()).hexdigest() != original['source_code_sha256']:
                raise ValueError('Original code bytes differ from source metadata')
            marker = 'Signature/context: ' + header['original_signature'] + '\n\n'
            if marker not in original['prompt']: raise ValueError('Original signature field differs')
            prompt = original['prompt'].replace(marker, 'Signature/context: ' + header['complete_header'] + '\n\n', 1)
            key = hashlib.sha256((original['id'] + '\nfull-source-header-v1\n' + prompt + '\n' + original['response']).encode()).hexdigest()
            row = {**original, 'id': key, 'prompt': prompt, 'split': 'train',
                'parent_instruction_id': original['id'],
                'parent_prompt_sha256': hashlib.sha256(original['prompt'].encode()).hexdigest(),
                'parent_response_sha256': hashlib.sha256(original['response'].encode()).hexdigest(),
                'task': 'complete_documented_code_header', 'complete_source_header': header['complete_header'],
                'header_end_byte': header['header_end_byte'], 'parser_node_type': header['node_type'],
                'original_header_text_cut': header['header_text_cut'],
                'original_method_body_in_signature': header['method_body_text_in_context'],
                'header_origin': 'Exact parsed original Go function/Ruby method header; source code response and metadata unchanged.',
                'external_model_teacher_used': False}
            inputs, labels = encoder.sequence(row, 'response')
            prompt_ids = encoder.tokens(encoder.prompt(row)); response_ids = encoder.tokens(row['response'])
            if inputs != [encoder.bos] + prompt_ids + response_ids + [encoder.eos]:
                exclusions.append({'id': key, 'parent_instruction_id': original['id'], 'language': language,
                    'reason': 'Complete corrected input would exceed actual Encoder allocation',
                    'prompt_tokens': len(prompt_ids), 'response_tokens': len(response_ids)})
                continue
            if labels != [-100] * (len(prompt_ids) + 1) + response_ids + [encoder.eos]:
                raise ValueError('Actual response-only target mask changed')
            if key in seen: raise ValueError('Repeated reconstructed header variant')
            seen.add(key); row['actual_target_tokens'] = len(response_ids) + 1
            row['complete_input_sha256'] = hashlib.sha256(json.dumps(inputs, separators=(',', ':')).encode()).hexdigest()
            destination.write(json.dumps(row, ensure_ascii=True) + '\n')
            retained_counts[language] += 1; targets += row['actual_target_tokens']; maximum = max(maximum, len(inputs))
    if candidate_counts != {language: stats['needs_correction'] for language, stats in audit['counts'].items()}:
        raise ValueError('Complete corrected header population differs from the audit')
    if file_hash(source) != digest or capture(files) != runtime: raise ValueError('Original source/grammar/runtime changed during preparation')
    manifest = {'prepared_at': datetime.now(timezone.utc).isoformat(), 'train_examples': sum(retained_counts.values()),
        'candidates_by_language': dict(candidate_counts), 'retained_by_language': dict(retained_counts),
        'excluded': exclusions, 'targets_per_pass': targets, 'maximum_complete_input_tokens': maximum,
        'source_data_sha256': digest, 'source_header_audit_sha256': file_hash(audit_path),
        'grammar_toolchain_manifest_sha256': parser.manifest_sha, 'tokenizer_sha256': file_hash(tokenizer_path),
        'config_sha256': file_hash(config_path), 'files_sha256': {'train.jsonl': file_hash(output)},
        'preparation_provenance': runtime, 'original_targets_and_source_metadata_preserved': True,
        'all_retained_actual_inputs_and_target_masks_complete': True, 'source_programs_executed': False,
        'original_training_data_changed': False, 'training_admitted': False, 'trained': False, 'goal_achieved': False,
        'limitations': ['Original code may depend on surrounding modules; parsing does not verify algorithmic behavior.',
            'Six Ruby source/context shapes unsupported by the audit are explicit and are not rewritten.',
            'Source-based variants are not new independent problems or measured model learning.',
            'Training admission still requires independent original-parent and held-group reconstruction.']}
    atomic_json(directory / 'manifest.json', manifest)
    print(json.dumps({'train_examples': manifest['train_examples'], 'languages': dict(retained_counts),
        'excluded': len(exclusions), 'targets_per_pass': targets, 'maximum_input_tokens': maximum}), flush=True)


if __name__ == '__main__': main()
