"""Preserve all128 exposed public programs for a future rehearsal component."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import random

from tokenizers import Tokenizer

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation_provenance import capture
from slm_lab.instruction_generalization_trial_v3 import complete_encoding
from slm_lab.instruction_trial_comparison import same, COMMON
from slm_lab.model import ModelConfig
from slm_lab.parameter_binding_curriculum_v1 import check
from slm_lab.training import Encoder

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'data/seen-code-retention-v1'
    if output.exists(): raise ValueError('Preserve earlier retention source generation')
    source = ROOT / 'checkpoints/instruction-generalization-probe-v3/data/verified_python.jsonl'
    before_path = ROOT / 'reports/exposed-code-learning-after-instruction-v1/after.json'
    after_path = ROOT / 'reports/semantic-code-optimization-v1/control/after-seen.json'
    tokenizer = ROOT / 'checkpoints/instruction-generalization-probe-v3/candidate/tokenizer.json'
    config = ROOT / 'configs/model.json'
    paths = [source, source.parent / 'manifest.json', before_path, after_path, tokenizer, config, Path(__file__),
             ROOT / 'src/slm_lab/parameter_binding_curriculum_v1.py',
             ROOT / 'src/slm_lab/instruction_generalization_trial_v3.py']
    paths += [ROOT / name for name in capture()['files_sha256']]
    paths = list(dict.fromkeys(paths)); runtime = capture(paths)
    before, after = [json.loads(p.read_text()) for p in [before_path, after_path]]
    same(before, after, COMMON + ['data_sha256', 'actual_exposures_sha256', 'selection', 'generation_budget'])
    identities = ['id', 'function', 'prompt_sha256', 'original_program_sha256', 'published_cases', 'target_tokens']
    if [[r[k] for k in identities] for r in before['results']] != [[r[k] for k in identities] for r in after['results']]:
        raise ValueError('Observed source population differs')
    if before['examples'] != 128 or after['examples'] != 128 or len(after['results']) != 128:
        raise ValueError('Complete original128-program population required')
    if file_hash(source) != before['data_sha256'] or file_hash(tokenizer) != before['tokenizer_sha256']:
        raise ValueError('Exact admitted source/tokenizer required')
    manifest = json.loads(paths[1].read_text())
    if file_hash(source) != manifest['files_sha256'][source.name]: raise ValueError('Source admission differs')
    lookup = {r['id']: r for r in jsonl(source)}
    rows = [lookup[r['id']] for r in before['results']]
    if len({r['original_program_sha256'] for r in rows}) != 128 or any(r['variant_index'] != 0 for r in rows):
        raise ValueError('Unique original-name training programs required')
    encoder = Encoder(Tokenizer.from_file(str(tokenizer)), ModelConfig.load(config), source.parent)
    counts = [complete_encoding(row, encoder) for row in rows]
    with ThreadPoolExecutor(max_workers=4) as pool: outcomes = list(pool.map(check, rows))
    for row, outcome, observed, count in zip(rows, outcomes, before['results'], counts):
        if (not outcome.get('functional_passed') or outcome['passed_cases'] != len(row['verification_cases'])
                or outcome['total_cases'] != len(row['verification_cases'])
                or len(row['verification_cases']) != observed['published_cases']
                or count['target_tokens'] != observed['target_tokens']):
            raise ValueError('Public program/example/target identity or execution differs')
    if capture(paths) != runtime or encoder.truncated: raise ValueError('Bound preparation changed or source clipped')
    output.mkdir()
    with (output / 'train.jsonl').open('w') as stream:
        for row in rows: stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    verification = [{'id': row['id'], **count, 'original_public_example_outcome': outcome}
                    for row, count, outcome in zip(rows, counts, outcomes)]
    with (output / 'verification.jsonl').open('w') as stream:
        for row in verification: stream.write(json.dumps(row) + '\n')
    order = list(range(128)); random.Random(88126).shuffle(order)
    exposures = [{'update': step + 1, 'index': order[step % 128],
                 'id': rows[order[step % 128]]['id'], **counts[order[step % 128]]} for step in range(1500)]
    with (output / 'proposed-exposures.jsonl').open('w') as stream:
        for row in exposures: stream.write(json.dumps(row) + '\n')
    previously_correct = [r['id'] for r in before['results'] if r['candidate']['functional_passed']]
    lost = [a['id'] for b, a in zip(before['results'], after['results'])
            if b['candidate']['functional_passed'] and not a['candidate']['functional_passed']]
    receipt = {'verified_at': datetime.now(timezone.utc).isoformat(), 'preparation_runtime': runtime,
        'source_rows': 128, 'source_rows_unchanged': True, 'all_original_public_examples_reexecuted': True,
        'previously_correct_source_ids': previously_correct, 'lost_after_narrow_supervision': lost,
        'complete_unique_response_targets': sum(r['target_tokens'] for r in counts),
        'proposed_updates': 1500, 'proposed_additional_response_targets': sum(r['target_tokens'] for r in exposures),
        'proposed_rehearsal_nll_weight': 0.25,
        'maximum_complete_context': max(r['context_tokens'] for r in counts),
        'files_sha256': {p.name: file_hash(p) for p in output.glob('*.jsonl')},
        'training_started': False, 'live_controller_sources_changed': False, 'human_preference_labels': False,
        'goal_achieved': False,
        'scope': 'Future original-public-program rehearsal and forgetting diagnostic. All128 programs were already used in training and shown in source diagnostics. No independent benchmark or learned retention benefit.'}
    atomic_json(output / 'manifest.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in {'preparation_runtime', 'previously_correct_source_ids', 'lost_after_narrow_supervision'}}, indent=2))


if __name__ == '__main__': main()
