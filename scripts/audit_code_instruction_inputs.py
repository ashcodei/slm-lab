"""Read every frozen code instruction through the actual CPU training Encoder."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

import torch
from tokenizers import Tokenizer

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.model import ModelConfig
from slm_lab.training import Encoder

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r'\A```([a-z]+)\n(.*)\n```\Z', re.S)


class RecordedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = []

    def token_to_id(self, value): return self.tokenizer.token_to_id(value)
    def get_vocab_size(self): return self.tokenizer.get_vocab_size()

    def encode(self, value):
        result = self.tokenizer.encode(value)
        self.calls.append((value, result.ids))
        return result


def audit(output):
    torch.set_num_threads(1)
    output = Path(output)
    if output.exists(): raise ValueError('Preserve an existing completed instruction audit')
    plan_path = ROOT / 'reports/extended-curriculum-v1/plan.json'
    plan = json.loads(plan_path.read_text())
    stage = next(item for item in plan['stages'] if item['name'] == 'multilingual-code-instruction')
    data = Path(stage['data'])
    data_sha = file_hash(data)
    if data_sha != stage['data_sha256']: raise ValueError('Frozen upcoming code instruction file changed')
    config_path = ROOT / 'configs/model.json'
    config = ModelConfig(**json.loads(config_path.read_text()))
    tokenizer_path = ROOT / 'checkpoints/extended-curriculum-v1/foundation-warmup/tokenizer.json'
    tokenizer_sha = file_hash(tokenizer_path)
    corpus = json.loads((ROOT / 'data/code-search-expanded-v1/corpus.json').read_text())
    if tokenizer_sha != corpus['tokenizer_sha256']: raise ValueError('Own code tokenizer differs from frozen source')
    recorder = RecordedTokenizer(Tokenizer.from_file(str(tokenizer_path)))
    encoder = Encoder(recorder, config, data.parent, stage['image_size'])
    files = [ROOT / name for name in capture()['files_sha256']]
    files += [Path(__file__), config_path, tokenizer_path]
    runtime = capture(files)
    totals = Counter(); groups = set(); ids = set(); language = defaultdict(Counter)
    examples = defaultdict(list)
    started = time.monotonic()
    progress = output.with_suffix('.progress.json')
    with data.open() as source:
        for index, line in enumerate(source, 1):
            row = json.loads(line)
            if row.get('image'): raise ValueError('Text code-instruction stage unexpectedly contains an image')
            bucket = row.get('language', 'general_human_instructions')
            counts = language[bucket]
            recorder.calls = []
            inputs, labels = encoder.sequence(row, 'response')
            if len(recorder.calls) != 2: raise ValueError('Unexpected actual Encoder tokenization calls')
            (prompt, prompt_ids), (_, response_ids) = recorder.calls
            if prompt != encoder.prompt(row) or recorder.calls[1][0] != row['response']:
                raise ValueError('Actual Encoder input differs from the frozen row')
            boundary = next(i for i, label in enumerate(labels) if label != -100)
            if (inputs[0] != encoder.bos or inputs[-1] != encoder.eos or labels != [-100] * boundary + inputs[boundary:]
                    or len(inputs) > config.max_text_length):
                raise ValueError('Actual BOS/EOS, response-only loss mask or length bound differs')
            prompt_cut = inputs[1:boundary] != prompt_ids
            response_cut = inputs[boundary:-1] != response_ids
            if not prompt_cut and not response_cut and inputs != [encoder.bos] + prompt_ids + response_ids + [encoder.eos]:
                raise ValueError('Complete actual Encoder sequence differs from the source tokens')
            for counter in (totals, counts):
                counter['rows'] += 1
                counter['full_prompt_tokens'] += len(prompt_ids)
                counter['full_response_tokens'] += len(response_ids)
                counter['actual_loss_targets_including_eos'] += len(inputs) - boundary
                counter['prompt_truncated_rows'] += int(prompt_cut)
                counter['response_truncated_rows'] += int(response_cut)
                counter['prompt_or_response_truncated_rows'] += int(prompt_cut or response_cut)
            if prompt_cut or response_cut:
                if len(examples['truncated']) < 20:
                    examples['truncated'].append({'id': row['id'], 'source': row['source'], 'language': bucket,
                        'prompt_tokens': len(prompt_ids), 'response_tokens': len(response_ids),
                        'retained_prompt_tokens': boundary - 1, 'retained_response_tokens': len(inputs) - boundary - 1,
                        'response_truncated': response_cut})
            if row['id'] in ids: totals['repeated_row_ids'] += 1
            ids.add(row['id']); groups.add(row.get('group_id', row['id']))
            if row['source'] == 'CodeSearchNet':
                match = FENCE.fullmatch(row['response'])
                if match is None: raise ValueError('Original code target is not a complete fenced source string')
                _, code = match.groups()
                if bucket != 'Python':
                    prefix = code.split('{', 1)[0][:500]
                    if 'Signature/context: ' + prefix + '\n\n' not in row['prompt']:
                        raise ValueError('Non-Python instruction signature differs from its known source transform')
                    flags = {'signature_prefix_at_500_character_cap': len(prefix) == 500,
                             'signature_prefix_raw_open_parentheses_exceed_closes': prefix.count('(') > prefix.count(')')}
                    if bucket == 'Ruby':
                        flags['ruby_signature_prefix_has_multiple_source_lines'] = '\n' in prefix.rstrip()
                    for key, value in flags.items():
                        totals[key] += int(value); counts[key] += int(value)
                        if value and len(examples[key]) < 10:
                            examples[key].append({'id': row['id'], 'source_url': row['source_url'], 'language': bucket,
                                'signature_context': prefix, 'original_target_prefix': code[:600]})
            if index % 10000 == 0:
                atomic_json(progress, {'updated_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
                    'rows': index, 'elapsed_seconds': time.monotonic() - started, 'totals': dict(totals),
                    'goal_achieved': False, 'status': 'reading frozen upcoming instructions through actual Encoder'})
                print('Actual code instruction rows checked', index, flush=True)
    if file_hash(data) != data_sha or capture(files) != runtime:
        raise ValueError('Frozen inputs or actual Encoder/runtime changed during audit')
    if encoder.truncated != totals['prompt_or_response_truncated_rows']:
        raise ValueError('Actual Encoder truncation ledger disagrees with direct input comparison')
    result = {'completed_at': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic() - started,
        'data': str(data), 'data_sha256': data_sha, 'plan_sha256': file_hash(plan_path), 'stage': stage,
        'config_sha256': file_hash(config_path), 'tokenizer_sha256': tokenizer_sha, 'evaluation_provenance': runtime,
        'actual_encoder_response_limit': max(1, config.max_text_length // 2 - 1), 'totals': dict(totals),
        'unique_row_ids': len(ids), 'source_groups': len(groups), 'by_language': {k: dict(v) for k, v in language.items()},
        'examples': dict(examples), 'actual_bos_eos_and_response_only_masks_all_verified': True,
        'all_records_checked': True, 'training_data_changed': False, 'new_training_started': False,
        'goal_achieved': False,
        'limitations': ['Token/input preservation does not prove that source code matches the described task or works with its dependencies.',
            'Raw parentheses and a 500-character cap are format indicators, not parsed proof of an invalid interface.',
            'Ruby multi-line signature contexts include source lines beyond the first line; some method headers can themselves span lines.',
            'Every row is checked once; final shuffled batch remainders can repeat a few examples and require actual training receipts.',
            'Human authorship, semantic benchmark independence and algorithmic correctness are not established by this audit.']}
    atomic_json(output, result)
    print(json.dumps({'report': str(output), 'totals': dict(totals)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/code-instruction-input-audit-v1.json')
    audit(parser.parse_args().output)
