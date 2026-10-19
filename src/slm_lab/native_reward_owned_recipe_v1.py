"""Non-neural source admission after the exact complete original joint run."""
import hashlib
import json
from pathlib import Path

from .exact_waiting_process_retirement_v1 import terminal

FORMAT = 'own-full-current-native-human-reward-owned-recipe-v1'
TRAIN = 85786
HELD = 4642
OPTIONS = dict(seed=998171, steps=2681, batch_size=32, learning_rate=.001,
    betas=[.9, .95], weight_decay=.01, gradient_clip=1.)
FEATURE_OPTIONS = dict(token_budget=16384, maximum_sequences=16)
RESERVE = 40 * 1024**3
BUDGET = dict(required_scratch_free_bytes=48 * 1024**3, required_home_free_bytes=RESERVE + 128 * 1024**2,
    maximum_train_shard_bytes=384 * 1024**2, maximum_held_shard_bytes=32 * 1024**2,
    maximum_head_state_bytes=32 * 1024**2, maximum_held_report_bytes=16 * 1024**2,
    maximum_native_export_bytes=5 * 1024**3, maximum_metadata_bytes=64 * 1024**2,
    maximum_published_bytes=128 * 1024**2)


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def completed_joint(folder, expected_owner):
    folder = Path(folder).resolve(); waiter = read(folder / 'waiter-plan.json')
    if waiter['owner'] != expected_owner:
        raise ValueError('The exact original full joint owner must remain bound')
    if not terminal(expected_owner): return None
    required = ['complete.json', 'restored.json', 'plan.json', 'children.json', 'train/native-export.json',
        'matched-comparisons.json', 'matched-reasoning.json']
    if (folder / 'failure.json').exists() or any(not (folder / n).is_file() for n in required):
        raise ValueError('The exact original joint owner stopped without all complete training/export/assessment evidence')
    plan, done, restored, exported = [read(folder / n) for n in ['plan.json', 'complete.json', 'restored.json', 'train/native-export.json']]
    if (plan['owner'] != expected_owner or done.get('complete') is not True
            or any(done.get(k) is not True for k in [
                'all5362original_joint_updates_and_full_native_export_complete',
                'all47actual_four_L40_capacity_cases_and_private_replay_and_native_GPUcold_complete',
                'both_full_original_QA_AI2D_MBPP_human_words_code_broad_and_reserved_reasoning_assessments_complete'])
            or done.get('actual_native_values') != 984399690
            or restored.get('all_registered_point_actors_terminal') is not True
            or any(not terminal(row) for row in read(folder / 'children.json'))
            or exported.get('complete') is not True or exported.get('mode') != 'train'
            or exported.get('completed_updates') != 5362 or exported.get('actual_counts', {}).get('total_unique') != 984399690
            or exported.get('every_complete_native_parameter_and_text_photo_word_output_cold_load_bit_exact') is not True):
        raise ValueError('Every complete original joint update, all four capacities/colds and full assessments must remain')
    comparison = read(folder / 'matched-comparisons.json')
    if (comparison.get('complete') is not True or comparison.get('full_original_populations_and_all_image_conditions_preserved') is not True
            or not comparison.get('source_files_sha256')):
        raise ValueError('The actual full original before/after assessment comparison must remain complete')
    for p, digest in comparison['source_files_sha256'].items():
        if sha(p) != digest: raise ValueError('An actual original assessment result changed after its complete paired comparison')
    for endpoint in ['initial', 'joint']:
        base = folder / endpoint / 'assessment'
        for name, count in [('full-QA.json',16961), ('full-AI2D.json',3088), ('full-original-MBPP.json',600),
                ('held-preferences.json',4642), ('all-held-words.json',3429)]:
            path = base / 'standalone' / name
            if len(read(path).get('results', [])) != count:
                raise ValueError('Every original complete QA,AI2D,MBPP,human and word result must remain present')
        reasoning = base / 'all-reserved-reasoning.json'
        if len(read(reasoning).get('results', [])) != 289:
            raise ValueError('All289original reserved reasoning deliveries must remain present at both endpoints')
        required.append(str(reasoning.relative_to(folder)))
    for rank in range(4):
        cold_path = folder / 'train' / ('cold-' + str(rank) + '.json'); cold = read(cold_path)
        rank_path = folder / 'train' / ('rank-' + str(rank) + '.json'); trained = read(rank_path)
        if (cold.get('full_size_GPU_native_cold_passed') is not True or cold.get('physical_GPU') != rank
                or cold.get('actual_native_values') != 984399690
                or cold.get('actual_checkpoint_sha256') != exported['checkpoint_sha256']
                or trained.get('complete') is not True or trained.get('rank') != rank or trained.get('world') != 4
                or trained.get('actual_full_size_GPU_joint_updates') != 5362
                or trained.get('actual_complete_native_context_tokens') != 91100747
                or trained.get('actual_native_response_EOS_targets') != 34500161
                or trained.get('actual_separate_CTC_UTF8_bytes') != 644063):
            raise ValueError('Every actual final original native GPU cold must match the fixed reward initializer')
        required.extend(['train/cold-' + str(rank) + '.json', 'train/rank-' + str(rank) + '.json'])
    checkpoint = Path(exported['checkpoint']).resolve(); tokenizer = checkpoint.parent / 'tokenizer.json'
    if (checkpoint.is_symlink() or sha(checkpoint) != exported['checkpoint_sha256']
            or sha(tokenizer) != exported['tokenizer_sha256']):
        raise ValueError('The actual final whole joint native and its original tokenizer must remain exact')
    files = {str(folder / n): sha(folder / n) for n in ['waiter-plan.json', *required]}
    files.update(comparison['source_files_sha256'])
    files[str(checkpoint)] = exported['checkpoint_sha256']; files[str(tokenizer)] = exported['tokenizer_sha256']
    return dict(initializer=dict(checkpoint=str(checkpoint), checkpoint_sha256=exported['checkpoint_sha256'],
        tokenizer=str(tokenizer), tokenizer_sha256=exported['tokenizer_sha256'], sources=exported['sources']),
        source_files_sha256=files, all_original_joint_scopes_complete=True)


def prepare(root, *, joint, expected_owner, runtime_sources):
    root = Path(root).resolve(); complete = completed_joint(joint, expected_owner)
    if complete is None: return None
    data = root / 'data/whole-human-alignment-v2'; manifest = read(data / 'manifest.json'); admission = read(data / 'admission.json')
    if (manifest.get('full_original_responses_and_EOS_preserved') is not True
            or manifest.get('tokenizer_sha256') != complete['initializer']['tokenizer_sha256']
            or manifest['splits']['preferences.train']['examples'] != TRAIN
            or manifest['splits']['preferences.valid']['examples'] != HELD
            or admission.get('training_admitted') is not True or admission.get('original_held_gradients_forbidden') is not True
            or admission['source_manifest_sha256'] != sha(data / 'manifest.json')
            or admission['independent_readback_sha256'] != sha(data / 'independent-source-readback.json')):
        raise ValueError('All original85786train/4642held human rankings and the exact native vocabulary must remain admitted')
    files = dict(runtime_sources); files.update(complete['source_files_sha256'])
    for name in ['manifest.json', 'admission.json', 'independent-source-readback.json']:
        files[str(data / name)] = sha(data / name)
    for role in ['train', 'valid']:
        for suffix in ['.jsonl', '.tokens.u16']:
            path = data / ('preferences.' + role + suffix); digest = manifest['files_sha256'][path.name]
            if sha(path) != digest: raise ValueError('Every complete human index and original native token byte must match')
            files[str(path)] = digest
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Every exact completed input and new reward runtime byte must agree')
    return dict(format=FORMAT, initializer=complete['initializer'], source_files_sha256=files,
        exact_original_joint_owner=expected_owner, original_full_joint_and_prior_pipeline_preserved=True,
        train_pairs=TRAIN, held_pairs=HELD, original_train_context_tokens=manifest['splits']['preferences.train']['packed_context_tokens'],
        original_held_context_tokens=manifest['splits']['preferences.valid']['packed_context_tokens'],
        data=str(data), human_manifest_sha256=sha(data / 'manifest.json'), human_admission_sha256=sha(data / 'admission.json'),
        options=dict(OPTIONS), feature_options=dict(FEATURE_OPTIONS), storage_budget=dict(BUDGET),
        feature_precision='fp32', expected_native_values=984399690, expected_reward_head_values=198272,
        final_original_joint_native_is_fixed_before_reward_results=True, no_held_based_rate_or_checkpoint_selection=True,
        all_original_human_responses_and_source_roles_preserved=True, own_output_human_judgments_not_supplied_by_public_rankings=True,
        GPU_training_admitted=False, PPO_training_admitted=False, candidate_promoted=False, goal_achieved=False)
