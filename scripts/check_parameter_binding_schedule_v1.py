"""Verify a future mixed instruction schedule; apply no model updates."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from tokenizers import Tokenizer

from slm_lab.binding_first_instruction_schedule_v1 import schedule
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.instruction_generalization_trial_v3 import complete_encoding, LANGUAGES
from slm_lab.instruction_program_schedule_v4 import schedule as earlier_schedule
from slm_lab.model import ModelConfig
from slm_lab.training import Encoder

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / 'reports/parameter-binding-curriculum-v1'
    if (directory / 'schedule-readiness.json').exists():
        raise ValueError('Preserve the earlier proposed schedule')
    original = ROOT / 'checkpoints/instruction-generalization-probe-v3/data'
    supplement = ROOT / 'data/parameter-binding-curriculum-v1'
    source_manifest = json.loads((original / 'manifest.json').read_text())
    binding_manifest = json.loads((supplement / 'manifest.json').read_text())
    inputs = [original / (name + '.jsonl') for name in [*LANGUAGES, 'verified_python', 'human_instruction']]
    inputs += [original / 'manifest.json', supplement / 'manifest.json']
    inputs += [supplement / name for name in binding_manifest['files_sha256']]
    tokenizer = ROOT / 'checkpoints/instruction-generalization-probe-v3/candidate/tokenizer.json'
    config = ROOT / 'configs/model.json'
    inputs += [tokenizer, config, Path(__file__), ROOT / 'src/slm_lab/binding_first_instruction_schedule_v1.py',
               ROOT / 'src/slm_lab/instruction_program_schedule_v4.py',
               ROOT / 'src/slm_lab/instruction_generalization_trial_v3.py', ROOT / 'src/slm_lab/training.py']
    bound = {str(p.relative_to(ROOT)): file_hash(p) for p in inputs}
    for p in inputs[:8]:
        if file_hash(p) != source_manifest['files_sha256'][p.name]: raise ValueError('Original source pool differs')
    for name, digest in binding_manifest['files_sha256'].items():
        if file_hash(supplement / name) != digest: raise ValueError('Binding source preparation differs')
    if (file_hash(tokenizer) != binding_manifest['tokenizer_sha256']
            or file_hash(config) != binding_manifest['model_config_sha256']
            or file_hash(original / 'verified_python.jsonl') != binding_manifest['source_sha256']):
        raise ValueError('Original source/tokenizer/config differs from preparation')
    pools = {name: list(jsonl(original / (name + '.jsonl'))) for name in [*LANGUAGES, 'verified_python', 'human_instruction']}
    additions = list(jsonl(supplement / 'train.jsonl'))
    parents = {row['id']: row for row in pools['verified_python'] if row['variant_index'] == 0}
    for row in additions:
        parent = parents[row['parent_training_id']]
        for key in ['function', 'group_id', 'original_program_sha256', 'verification_cases', 'source', 'license', 'source_url']:
            if row[key] != parent[key]: raise ValueError('Supplement changes original source identity or public examples')
    pools['verified_python'] += additions
    encoder = Encoder(Tokenizer.from_file(str(tokenizer)), ModelConfig.load(config), original)
    proposed = schedule(pools, 1500, 88125)
    baseline = earlier_schedule(pools, 1500, 88125)
    # Module/human choices stay exactly matched to the same source schedule.
    if any([a[i] for i in [0, 1, 3]] != [b[i] for i in [0, 1, 3]] for a, b in zip(proposed, baseline)):
        raise ValueError('Existing module/human exposure changed')
    exposure = []; lookup = {}
    for step, group in enumerate(proposed, start=1):
        for pick in group:
            row = pools[pick['pool']][pick['index']]
            key = (pick['pool'], row['id'])
            if key not in lookup: lookup[key] = complete_encoding(row, encoder)
            exposure.append({'update': step, **pick, 'id': row['id'],
                'group_id': row.get('group_id'), 'original_program_sha256': row.get('original_program_sha256'),
                'variant_index': row.get('variant_index'), 'binding_variant': row.get('binding_variant', False),
                **lookup[key]})
    programs = {r['original_program_sha256'] for r in pools['verified_python']}
    selections = [pools['verified_python'][group[2]['index']] for group in proposed]
    first = selections[:len(programs)]
    if len({r['original_program_sha256'] for r in first}) != len(programs) or any(r['variant_index'] != 0 for r in first):
        raise ValueError('Every original algorithm must precede variants')
    selected_binding = [r for r in selections if r.get('binding_variant')]
    if {r['original_program_sha256'] for r in selected_binding} != {r['original_program_sha256'] for r in additions}:
        raise ValueError('Future budget leaves available argument-binding algorithms unvisited')
    if encoder.truncated or any(file_hash(ROOT / name) != digest for name, digest in bound.items()):
        raise ValueError('Target clipping or bound source mutation')
    path = directory / 'proposed-exposures.jsonl'
    with path.open('w') as stream:
        for row in exposure: stream.write(json.dumps(row) + '\n')
    report = {'verified_at': datetime.now(timezone.utc).isoformat(), 'bound_files_sha256': bound,
        'updates': 1500, 'example_exposures': len(exposure), 'source_pool_exposures': dict(Counter(r['pool'] for r in exposure)),
        'source_algorithms': len(programs), 'combined_verified_source_rows': len(pools['verified_python']),
        'all_originals_before_variants': True, 'selected_binding_exposures': len(selected_binding),
        'selected_distinct_binding_algorithms': len({r['original_program_sha256'] for r in selected_binding}),
        'binding_exposures_under_earlier_sorted_variant_schedule': sum(pools['verified_python'][g[2]['index']].get('binding_variant', False) for g in baseline),
        'actual_complete_response_targets': sum(r['target_tokens'] for r in exposure),
        'maximum_complete_context': max(r['context_tokens'] for r in exposure),
        'proposed_exposures_sha256': file_hash(path), 'module_and_human_choices_preserved': True,
        'source_program_file_groups_unchanged': True, 'training_started': False,
        'live_controller_sources_changed': False, 'goal_achieved': False,
        'scope': 'Future data/exposure readiness only. No learned binding, held correctness, multimodal retention or human quality established.'}
    atomic_json(directory / 'schedule-readiness.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'bound_files_sha256'}, indent=2))


if __name__ == '__main__': main()
