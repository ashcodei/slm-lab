"""Execution-derived preferences for runnable own code with wrong results."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from .code_quality import execute, instruction_code
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .evaluation_provenance import capture, ROOT
from .instruction_generalization_trial_v3 import complete_encoding


def outcome(result):
    """Compare deterministic correctness evidence, excluding measured timings."""
    return {key: result.get(key) for key in
            ['syntax_valid', 'functional_passed', 'passed_cases', 'total_cases', 'phase', 'error']}


def prepare(data, exposures, diagnostic, checkpoint, output, encoder):
    data, exposures, diagnostic, checkpoint, output = map(
        Path, [data, exposures, diagnostic, checkpoint, output])
    if output.exists():
        raise ValueError('Preserve existing semantic-preference preparation')
    source = json.loads((data.parent / 'manifest.json').read_text())
    generated = json.loads(diagnostic.read_text())
    bound_inputs = {path: file_hash(path) for path in
                    [data, exposures, data.parent / 'manifest.json', diagnostic, checkpoint, Path(__file__)]}
    digests = {path.name: file_hash(path) for path in [data, exposures]}
    if any(source['files_sha256'][name] != digest for name, digest in digests.items()):
        raise ValueError('Original admitted source or actual exposure plan changed')
    if (generated['data_sha256'] != digests[data.name]
            or generated['actual_exposures_sha256'] != digests[exposures.name]
            or generated['checkpoint_sha256'] != file_hash(checkpoint)
            or not generated['exact_language_and_critic_preserved']):
        raise ValueError('Own generation, model or source identity differs')
    runtime = generated['evaluation_provenance']
    bound_files = [ROOT / path for path in runtime['files_sha256']]
    if capture(bound_files) != runtime:
        raise ValueError('Exact original generation evaluator/runtime changed')
    originals = list(jsonl(data))
    source_by_id = {row['id']: row for row in originals}
    if len(source_by_id) != len(originals):
        raise ValueError('Duplicate admitted source IDs')
    seen = {row['id'] for row in jsonl(exposures) if row['pool'] == 'verified_python'}
    if generated['examples'] != len(generated['results']) or not generated['results']:
        raise ValueError('Incomplete or empty diagnostic population')
    pairs = []
    excluded = Counter()
    programs = set()
    for result in generated['results']:
        row = source_by_id.get(result['id'])
        if row is None or row['id'] not in seen or row['variant_index'] != 0:
            raise ValueError('Expected actually exposed original-name training programs')
        program = row['original_program_sha256']
        prompt_digest = hashlib.sha256(encoder.prompt(row).encode()).hexdigest()
        if (program != result['original_program_sha256'] or row['function'] != result['function']
                or prompt_digest != result['prompt_sha256']
                or len(row['verification_cases']) != result['published_cases']
                or program in programs):
            raise ValueError('Changed or repeated source algorithm, prompt or published examples')
        programs.add(program)
        chosen, rejected = row['response'], result['raw_response']
        counts = complete_encoding(row, encoder)
        if counts['target_tokens'] != result['target_tokens']:
            raise ValueError('Original complete diagnostic target differs')

        def check(text):
            return execute({'language': 'Python', 'code': instruction_code(text),
                            'function': row['function'], 'cases': row['verification_cases'], 'stress': []})

        gold, actual = check(chosen), check(rejected)
        if not gold.get('functional_passed'):
            raise ValueError('Original public preferred program fails its published examples')
        if outcome(actual) != outcome(result['candidate']):
            raise ValueError('Own generated failure is not reproducible')
        if actual.get('functional_passed'):
            excluded['own_generation_already_passes'] += 1
            continue
        if not actual.get('syntax_valid'):
            excluded['invalid_syntax'] += 1
            continue
        if actual.get('phase') != 'correctness':
            excluded['missing_callable_or_restricted_module'] += 1
            continue
        if actual.get('error') != 'Hidden case mismatch':
            excluded['runtime_error_or_timeout_instead_of_wrong_result'] += 1
            continue
        if not rejected.strip() or chosen == rejected:
            raise ValueError('Contradictory execution-derived preference')
        rejected_counts = complete_encoding({**row, 'response': rejected}, encoder)
        pairs.append({key: value for key, value in row.items()
                      if key not in {'response', 'annotation_origin'}} | {
            'chosen': chosen, 'rejected': rejected,
            'chosen_target_tokens': counts['target_tokens'],
            'rejected_target_tokens': rejected_counts['target_tokens'],
            'feedback_source': 'synthetic',
            'feedback_origin': 'Original public reference passes; runnable own generated code returns a wrong result on a published source example.',
            'own_generation_checkpoint_sha256': generated['checkpoint_sha256'],
            'own_generation_diagnostic_sha256': file_hash(diagnostic),
            'verification': {'preferred': gold, 'rejected': actual},
            'annotation_origin': 'Execution-derived training-source labels, not human ratings or external-model teaching.'})
    if encoder.truncated:
        raise ValueError('Semantic-preference source or own response was truncated')
    if capture(bound_files) != runtime or any(file_hash(path) != digest for path, digest in bound_inputs.items()):
        raise ValueError('Bound source/evaluator changed during preparation')
    output.mkdir(parents=True)
    (output / 'train.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in pairs))
    manifest = {
        'train_examples': len(pairs), 'unique_source_algorithms': len(pairs),
        'diagnostic_examples': len(programs), 'excluded': dict(excluded),
        'source_data_sha256': digests[data.name], 'actual_exposures_sha256': digests[exposures.name],
        'source_manifest_sha256': bound_inputs[data.parent / 'manifest.json'],
        'own_checkpoint_sha256': generated['checkpoint_sha256'],
        'own_diagnostic_sha256': bound_inputs[diagnostic], 'train_sha256': file_hash(output / 'train.jsonl'),
        'implementation_sha256': bound_inputs[Path(__file__)], 'generation_evaluation_runtime': runtime,
        'chosen_target_tokens': sum(row['chosen_target_tokens'] for row in pairs),
        'rejected_target_tokens': sum(row['rejected_target_tokens'] for row in pairs),
        'feedback_source': 'synthetic', 'human_feedback': False, 'external_teacher_used': False,
        'all_retained_generations_runnable_and_return_wrong_values': True,
        'all_retained_public_positives_pass': True, 'all_accepted_targets_and_masks_complete': True,
        'data_ready': bool(pairs), 'trained': False, 'automatic_policy_promotion': False,
        'scope': 'Own-policy wrong-result contrasts on actually exposed public training programs and visible examples. No held-out, efficiency or human-preference evidence.',
        'limitations': ['Published examples are incomplete specifications, not proofs of correct algorithms.',
                        'Selected training examples measure learning/memorization; independent final assessment remains required.',
                        'Syntax errors, missing/restricted callables and runtime failures are excluded from this narrow wrong-result contrast population.',
                        'Preference learning must be compared with a same-source supervised control before claiming its benefit.'],
        'goal_achieved': False}
    atomic_json(output / 'manifest.json', manifest)
    return manifest
