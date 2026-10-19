"""Native closed-book answers to all original NQ-open development questions."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import unicodedata

import torch
from tokenizers import Tokenizer

from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .evaluation_provenance import capture, seal
from .nq_input_audit import admit
from .training import Encoder, load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'All3610 original NQ-open development questions; question only,closed-book,greedy32tokens; NFKC/casefold/whitespace exact match to any unchanged original alias.'


def normalized(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def correct(generated, answers):
    return bool(normalized(generated)) and any(normalized(generated) == normalized(value) for value in answers)


def summarize(rows):
    def metrics(cases):
        total = len(cases); hits = sum(row['exact'] for row in cases)
        return {'examples': total, 'correct': hits, 'exact_accuracy': hits / total if total else None}
    return {'all_original_development': metrics(rows),
            'no_exact_question_match_in_audited_sources': metrics([row for row in rows if not row['potential_exact_question_exposure']]),
            'potential_exact_question_exposure': metrics([row for row in rows if row['potential_exact_question_exposure']])}


def compare_reports(before, after):
    for key in ['evaluation_provenance', 'data_sha256', 'manifest_sha256', 'tokenizer_sha256', 'exposure_audit_sha256',
                'input_audit_sha256', 'inference_precision', 'inference_device_type', 'generation_budget', 'protocol']:
        if before[key] != after[key]: raise ValueError('Unmatched native NQ assessment: ' + key)
    identities = []
    for report in [before, after]:
        rows = report['results']
        if len(rows) != 3610 or len({row['id'] for row in rows}) != 3610: raise ValueError('All3610 original NQ cases required')
        if report['overall'] != summarize(rows): raise ValueError('NQ aggregate differs from case evidence')
        for row in rows:
            if row['exact'] != correct(row['generated'], row['reference_answers']): raise ValueError('NQ score differs from actual generated answer/aliases')
        identities.append([(row['id'], row['group_id'], row['prompt_sha256'], row['prompt_tokens'], row['reference_answers'],
                            row['potential_exact_question_exposure']) for row in rows])
    if identities[0] != identities[1]: raise ValueError('Unmatched original NQ question/answer/prompt/exposure cases')
    lost = [first['id'] for first, second in zip(before['results'], after['results']) if first['exact'] and not second['exact']]
    return {'passed': not lost, 'previously_correct_questions_lost': lost,
            'before': before['overall'], 'after': after['overall'],
            'base_checkpoint_sha256': before['checkpoint_sha256'], 'candidate_checkpoint_sha256': after['checkpoint_sha256'],
            'goal_achieved': False, 'scope': 'Full matched original NQ-open development retention, with potential exact exposure explicit; not independent final broad capability.'}


@torch.inference_mode()
def evaluate(checkpoint, output, device='cuda'):
    output = Path(output)
    if output.exists(): raise ValueError('Preserve existing native NQ assessment')
    _, admission = admit(ROOT)
    data = ROOT / 'data/nq-human-questions-v1'; path = data / 'valid.jsonl'; manifest_path = data / 'manifest.json'
    data_sha = file_hash(path); manifest_sha = file_hash(manifest_path); checkpoint_sha = file_hash(checkpoint)
    manifest = json.loads(manifest_path.read_text())
    exposed = set(admission['development_cases_with_exact_potential_training_exposure'])
    files = [ROOT / name for name in capture()['files_sha256']] + [Path(__file__), ROOT / 'src/slm_lab/nq_input_audit.py',
             ROOT / 'src/slm_lab/nq_open_data.py', ROOT / 'scripts/fetch_nq_open_source.py', ROOT / 'scripts/audit_nq_exposure.py',
             ROOT / 'src/slm_lab/exposure_audit.py']
    provenance = capture(files); torch.set_num_threads(1)
    model, metadata = load_checkpoint(checkpoint, device, training_state=False); model.eval()
    tokenizer_path = Path(checkpoint).parent / 'tokenizer.json'
    if file_hash(tokenizer_path) != manifest['tokenizer_sha256'] or metadata['tokenizer_sha256'] != manifest['tokenizer_sha256']:
        raise ValueError('Own policy and original human QA tokenizer differ')
    encoder = Encoder(Tokenizer.from_file(str(tokenizer_path)), model.config, data, 448)
    rows = list(jsonl(path))
    if len(rows) != 3610 or not exposed <= {row['id'] for row in rows}: raise ValueError('Original NQ development/exposure population changed')
    results = []; amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    for row in rows:
        prompt = encoder.prompt(row); tokens = [encoder.bos] + encoder.tokens(prompt)
        if len(tokens) + 32 > model.config.max_text_length: raise ValueError('Complete original NQ question exceeds declared generation context')
        ids = torch.tensor([tokens], device=device)
        with amp(): generated = model.generate(ids, max_new_tokens=32, eos_token_id=encoder.eos)
        text = encoder.tokenizer.decode(generated[0, len(tokens):].tolist())
        results.append({'id': row['id'], 'group_id': row['group_id'], 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                        'prompt_tokens': len(tokens), 'reference_answers': row['reference_answers'],
                        'generated': text, 'exact': correct(text, row['reference_answers']),
                        'potential_exact_question_exposure': row['id'] in exposed})
        if len(results) % 100 == 0: print('Original native NQ questions actually generated', len(results), len(rows), flush=True)
    if file_hash(checkpoint) != checkpoint_sha or file_hash(manifest_path) != manifest_sha:
        raise ValueError('Own NQ policy/data changed during generation')
    for name, digest in [('nq-original-input-audit-v1.json', admission['input_audit_sha256']),
                         ('nq-development-exposure-v1/audit.json', admission['exposure_audit_sha256'])]:
        if file_hash(ROOT / 'reports' / name) != digest: raise ValueError('NQ input/exposure evidence changed during generation')
    report = {'checkpoint_sha256': checkpoint_sha, 'data_sha256': data_sha, 'manifest_sha256': manifest_sha,
              'tokenizer_sha256': metadata['tokenizer_sha256'], 'generation_budget': 32, 'protocol': PROTOCOL,
              'input_audit_sha256': admission['input_audit_sha256'], 'exposure_audit_sha256': admission['exposure_audit_sha256'],
              'results': results, 'overall': summarize(results), 'goal_achieved': False,
              'limitations': ['Strict local alias match does not implement full official article/punctuation or token-F1 scoring.',
                              'No-exact-question-match cases can still have answer/concept/paraphrase/article overlap; they are not certified independently unseen.',
                              'Original development split is a tuning assessment; this does not certify all-domain knowledge or frontier parity.']}
    seal(report, provenance, files, data=path, digest=data_sha, device=device, tokenizer_sha256=metadata['tokenizer_sha256'])
    atomic_json(output, report); return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.output, args.device)['overall']), flush=True)
