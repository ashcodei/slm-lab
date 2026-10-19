"""Learn additional real photos and reviewed dialogues with full rehearsal before RLHF."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from slm_lab.alignment_selection import REPORT_FILES, compare, validate_collection
from slm_lab.capability_protocol import specifications
from slm_lab.completed_state_cleanup import retire
from slm_lab.exercise_evaluation import compare_reports as compare_exercises
from slm_lab.exercise_cpp_selection import compare_reports as compare_cpp_exercises
from slm_lab.exercise_js_evaluation import compare_reports as compare_js_exercises
from slm_lab.exercise_go_evaluation import compare_reports as compare_go_exercises
from slm_lab.exercise_rust_evaluation_v2 import compare_reports as compare_rust_exercises
from slm_lab.feedback_exercise_rust_guard import validate_coverage as validate_rust_coverage
from slm_lab.corpus import file_hash
from slm_lab.corpus import PackedCorpus
from slm_lab.code_continuation_receipt import verify_targets
from slm_lab.clevr_evaluation import compare_reports as compare_visual_reasoning
from slm_lab.textvqa_admission import admit as admit_real_reading
from slm_lab.textvqa_evaluation import compare_reports as compare_real_reading
from slm_lab.own_ocr_question_admission import admit as admit_own_context_reading
from slm_lab.own_context_transfer_evaluation import compare_reports as compare_own_context_reading
from slm_lab.nq_input_audit import admit as admit_human_fact_questions
from slm_lab.nq_open_evaluation import compare_reports as compare_human_fact_questions
from slm_lab.complete_js_parameter_admission import admit as admit_complete_js_parameters
from slm_lab.goruby_header_admission import admit as admit_goruby_headers
from slm_lab.vqa_pair_admission import admit as admit_paired_photos
from slm_lab.vqa_pair_guard import compare as compare_paired_photos, validate_coverage as validate_paired_photos
from slm_lab.systems_variety_training_admission import admit as admit_systems_variety
from slm_lab.exercism_support_admission import admit as admit_exercise_support_context
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation_provenance import capture
from slm_lab.masked_visual_pretraining import process_identity
from slm_lab.vision_transfer import language_digest
import torch
from train_multimodal_pilot import complete_stage
from prepare_broad_code_continuation import verify_packed

ROOT = Path(__file__).resolve().parents[1]
DONE = 'human dialogue retention training/evaluation completed; broader capability unverified'


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preceding-pid', type=int, required=True)
    parser.add_argument('--photos-pid', type=int, required=True)
    parser.add_argument('--words-pid', type=int, required=True)
    parser.add_argument('--clevr-pid', type=int, required=True)
    parser.add_argument('--vqa-pairs-pid', type=int, required=True)
    args = parser.parse_args()
    reports = ROOT / 'reports/dialogue-retention-v1'; reports.mkdir(exist_ok=True)
    checkpoints = ROOT / 'checkpoints/dialogue-retention-v1'
    files = [Path(__file__), *[ROOT / 'src/slm_lab' / (name + '.py') for name in
             ['model', 'training', 'objectives', 'alignment_selection', 'evaluation', 'evaluation_provenance',
              'masked_visual_pretraining', 'capability_protocol', 'expanded_photos', 'expanded_scene_words',
              'vision_transfer', 'completed_state_cleanup', 'exercise_code', 'exercise_evaluation',
              'exercise_cpp_evaluation', 'exercise_cpp_runner', 'exercise_cpp_selection', 'native_code',
              'exercise_js_evaluation', 'exercise_js_runner',
              'exercise_go_evaluation', 'exercise_go_runner',
              'exercise_rust_evaluation_v2', 'exercise_rust_runner', 'feedback_exercise_rust_guard',
              'corpus', 'systems_code', 'mix_corpora', 'code_continuation_receipt',
              'clevr_data', 'clevr_programs', 'remote_zip', 'clevr_evaluation',
              'textvqa_data', 'textvqa_input_audit', 'textvqa_admission', 'textvqa_evaluation',
              'vqa_pairs', 'vqa_attribution', 'vqa_pair_evaluation', 'vqa_pair_admission', 'vqa_pair_guard',
              'own_ocr_question_admission', 'own_ocr_training_data',
              'own_ocr_context', 'own_ocr_question_evaluation', 'own_context_transfer_evaluation',
              'photo_word_reading', 'word_localizer', 'byte_word_reader', 'text_localization_metrics',
              'nq_input_audit', 'nq_open_data', 'nq_open_evaluation', 'exposure_audit', 'complete_js_parameter_admission',
              'goruby_code_headers', 'goruby_header_admission',
              'systems_variety', 'systems_variety_admission', 'systems_variety_training_admission']]]
    files.append(ROOT / 'scripts/native_sandbox.py')
    files.append(ROOT / 'scripts/javascript_sandbox.py')
    files.append(ROOT / 'scripts/verify_exercise_js_references.py')
    files.append(ROOT / '.tools/javascript-toolchain-v1.json')
    files.append(ROOT / '.tools/javascript-package-lock-v1.json')
    files.append(ROOT / 'reports/exercism-javascript-publisher-references-v1.json')
    files.append(ROOT / 'scripts/go_sandbox.py')
    files.append(ROOT / 'scripts/verify_exercise_go_references.py')
    files.append(ROOT / '.tools/go-toolchain-v1.json')
    files.append(ROOT / 'reports/exercism-go-publisher-references-v1.json')
    files.extend(ROOT / name for name in ['scripts/rust_sandbox.py', 'scripts/verify_exercise_rust_references.py',
        'scripts/verify_rust_confinement.py', 'scripts/prepare_rust_test_libraries.py', '.tools/rust-toolchain-v1.json',
        'reports/exercism-rust-publisher-references-v2.json', 'reports/rust-kernel-and-report-verification-v1.json'])
    files.append(ROOT / 'scripts/prepare_broad_code_continuation.py')
    files.append(ROOT / 'scripts/audit_clevr_inputs.py')
    files.append(ROOT / 'scripts/fetch_nq_open_source.py')
    files.append(ROOT / 'scripts/audit_nq_exposure.py')
    files.append(ROOT / 'scripts/js_instruction_header.cjs')
    files.append(ROOT / 'scripts/prepare_js_parameter_variants.cjs')
    files.append(ROOT / 'scripts/audit_js_instruction_interfaces.cjs')
    files.append(ROOT / 'data/complete-js-parameters-v1/manifest.json')
    files.append(ROOT / 'reports/complete-js-parameter-variant-reconstruction-v1.json')
    files.append(ROOT / 'scripts/audit_goruby_instruction_headers.py')
    files.append(ROOT / 'scripts/prepare_goruby_header_variants.py')
    files.append(ROOT / 'scripts/verify_goruby_header_parser.py')
    files.append(ROOT / '.tools/goruby-parser-v1.json')
    files.append(ROOT / 'data/complete-goruby-headers-v1/manifest.json')
    files.append(ROOT / 'configs/systems-variety-sources-v1.json')
    files.append(ROOT / 'scripts/prepare_systems_variety_code.py')
    files.append(ROOT / 'data/public-systems-variety-v1/manifest.json')
    files.append(ROOT / 'data/systems-variety-code-continuation-v1/manifest.json')
    files.extend(ROOT / name for name in [
        'src/slm_lab/exercism_support_admission.py', 'src/slm_lab/instruction_generalization_trial.py',
        'scripts/prepare_exercism_support_context_v3.py', 'data/exercism-support-context-v3/manifest.json',
        'data/exercism-support-context-v3/train.jsonl', 'reports/exercism-support-context-v3-readiness.json',
        'reports/training-source-native-quality-v2/lens-person-relocated-v3.json', 'configs/model.json'])
    loaded = capture(files)

    def status(message, **fields):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
                    'goal_achieved': False, **fields})

    def run(command):
        if capture(files) != loaded:
            raise ValueError('Loaded dialogue-retention implementation/runtime changed; inspect/reload waiting scheduler')
        subprocess.run(command, check=True, cwd=ROOT, env={**os.environ, 'OMP_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'})

    def capabilities(directory):
        return {name: json.loads((directory / path).read_text()) for name, path in REPORT_FILES.items()}

    try:
        pid = args.preceding_pid; identity = process_identity(pid)
        if identity is None:
            raise ValueError('Preceding curriculum PID is not live; inspect completed artifacts before scheduling')
        arguments = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if not any(part.endswith(b'scripts/train_extended_curriculum.py') for part in arguments):
            raise ValueError('Preceding process is not the existing extended curriculum')
        photos = ROOT / 'data/expanded-everyday-photos-v1'
        photo_identity = process_identity(args.photos_pid)
        if not (photos / 'manifest.json').exists() and (photo_identity is None or b'slm_lab.expanded_photos' not in
                Path(f'/proc/{args.photos_pid}/cmdline').read_bytes().split(b'\0')):
            raise ValueError('Additional-photo collector is not verified live or finalized')
        words = ROOT / 'data/expanded-scene-words-v1'
        words_identity = process_identity(args.words_pid)
        if not (words / 'manifest.json').exists() and (words_identity is None or b'slm_lab.expanded_scene_words' not in
                Path(f'/proc/{args.words_pid}/cmdline').read_bytes().split(b'\0')):
            raise ValueError('Additional scene-word preparation is not verified live or finalized')
        clevr = ROOT / 'data/clevr-reasoning-v1'; clevr_identity = process_identity(args.clevr_pid)
        if not (clevr / 'manifest.json').exists() and (clevr_identity is None or b'slm_lab.clevr_data' not in
                Path(f'/proc/{args.clevr_pid}/cmdline').read_bytes().split(b'\0')):
            raise ValueError('Original CLEVR preparation is not verified live or finalized')
        paired_photo_data = ROOT / 'data/vqa-real-pairs-v1'
        paired_photo_identity = process_identity(args.vqa_pairs_pid)
        if not (paired_photo_data / 'manifest.json').exists() and (paired_photo_identity is None or b'slm_lab.vqa_pairs' not in
                Path(f'/proc/{args.vqa_pairs_pid}/cmdline').read_bytes().split(b'\0')):
            raise ValueError('Original paired-photo collector is not verified live or finalized')
        status('waiting for verified live extended curriculum', preceding_pid=pid, process_start_ticks=identity,
               photos_pid=args.photos_pid, photos_start_ticks=photo_identity,
               words_pid=args.words_pid, words_start_ticks=words_identity,
               clevr_pid=args.clevr_pid, clevr_start_ticks=clevr_identity,
               vqa_pairs_pid=args.vqa_pairs_pid, vqa_pairs_start_ticks=paired_photo_identity,
               next_training='Broad code continuation, then original OS/network/security code with complete earlier code rehearsal, real-word warmup and joint photos/paired real-photo questions/plain and own-context reading/human fact questions/visual reasoning/dialogues/full rehearsal; finalized original photo pairs and NQ exposure audit required')
        while process_identity(pid) == identity:
            time.sleep(15)
        if capture(files) != loaded:
            raise ValueError('Dialogue-retention source/runtime changed during preceding training')
        preceding_reports = ROOT / 'reports/extended-curriculum-v1'
        preceding = json.loads((preceding_reports / 'status.json').read_text())
        if preceding['status'] != 'extended language and multimodal training/evaluation completed; broader capability unverified':
            raise ValueError('Preceding curriculum did not finish its declared training/evaluation')
        base = ROOT / 'checkpoints/extended-curriculum-v1/joint/model.pt'
        # Some controllers emit both code modes from one CLI invocation.
        # Fill only genuinely absent artifacts before validating all nine families.
        for family, module, name, extra in specifications(ROOT):
            if not (preceding_reports / name).exists():
                status('completing missing preceding capability assessment', family=family)
                run([sys.executable, '-u', '-m', module, '--checkpoint', str(base), '--output', str(preceding_reports / name),
                     '--device', 'cuda', *extra])
        baseline = capabilities(preceding_reports); base_sha = file_hash(base)
        if (validate_collection(baseline, list(REPORT_FILES)) != base_sha
                or baseline['code']['evaluation_provenance'] != capture()):
            raise ValueError('Completed curriculum capability evidence differs from its frozen base/current evaluator')
        if (base.parent / 'training-state.pt').exists():
            # The main controller has exited and all nine assessments now match
            # retained final weights. Release only its redundant optimizer state
            # before waiting for a collector that can pause for free disk.
            retire(base.parent, reports / 'completed-base-state-cleanup.json')
        broad_root = ROOT / 'data/broad-code-continuation-v1'
        broad = json.loads((broad_root / 'manifest.json').read_text())
        if (file_hash(broad_root / 'packed/corpus.json') != broad['packed_manifest_sha256'] or
            file_hash(ROOT / 'scripts/prepare_broad_code_continuation.py') != broad['preparation_source_sha256']):
            raise ValueError('Prepared original broad-code corpus/implementation changed')
        code_corpus = verify_packed(broad_root / 'packed', broad['tokenizer_sha256'])
        source_metadata = torch.load(base, map_location='cpu', weights_only=True, mmap=True)
        if source_metadata['tokenizer_sha256'] != broad['tokenizer_sha256']:
            raise ValueError('Original library-code tokenizer differs from completed main policy')
        context = source_metadata['config']['max_text_length']
        del source_metadata
        windows = PackedCorpus(broad_root / 'packed', broad['tokenizer_sha256'], context)
        code_steps = math.ceil(len(windows) / 64)
        code_directory = checkpoints / 'broad-code'
        code_spec = {'stage': 'pretrain', 'steps': code_steps, 'trainable': 'language', 'image_size': 448,
                     'data_sha256': broad['packed_manifest_sha256']}
        atomic_json(reports / 'broad-code-plan.json', {**code_spec, 'base_checkpoint_sha256': base_sha,
                    'corpus_manifest_sha256': file_hash(broad_root / 'manifest.json'),
                    'prior_code_train_tokens_rehearsed': broad['old_train_tokens_rehearsed'],
                    'new_library_train_tokens': broad['new_train_tokens'], 'train_tokens': code_corpus['splits']['train']['tokens'],
                    'world_size': 4, 'microbatch': 1, 'gradient_accumulation': 16, 'learning_rate': 2e-5,
                    'nominal_target_exposures': code_steps * 64 * (context - 1),
                    'packed_windows': len(windows),
                    'packed_tail_token_positions_not_sampled': code_corpus['splits']['train']['tokens'] - 1 - len(windows) * (context - 1),
                    'final_batch_repeated_windows': code_steps * 64 - len(windows),
                    'scope': 'One pass plus final batch remainder; own language only, before supervised instruction/image/dialogue rehearsal.',
                    'goal_achieved': False})
        if not complete_stage(code_directory, code_spec):
            if code_directory.exists() and any(code_directory.iterdir()):
                raise ValueError('Incomplete broad-code continuation requires inspected resume')
            if shutil.disk_usage(ROOT).free < 25 * 1024**3:
                raise ValueError('Insufficient free disk for broad-code continuation')
            status('training original library code with prior broad-code rehearsal', steps=code_steps,
                   new_library_tokens=broad['new_train_tokens'], prior_code_tokens=broad['old_train_tokens_rehearsed'])
            run([str(ROOT / '.venv/bin/torchrun'), '--standalone', '--nproc_per_node=4', '-m', 'slm_lab.training',
                 'pretrain', '--checkpoint', str(base), '--tokenizer', str(base.parent / 'tokenizer.json'),
                 '--data', str(broad_root / 'packed'), '--output', str(code_directory), '--trainable', 'language',
                 '--steps', str(code_steps), '--batch-size', '1', '--gradient-accumulation', '16', '--learning-rate', '2e-5',
                 '--image-size', '448', '--device', 'cuda', '--threads', '1', '--gradient-checkpointing',
                 '--checkpoint-every', '250', '--keep-checkpoints', '1', '--log-every', '10'])
            if not complete_stage(code_directory, code_spec): raise ValueError('Broad-code continuation did not complete')
        target_receipt = verify_targets(json.loads((code_directory / 'training.json').read_text()), code_steps, context)
        atomic_json(reports / 'broad-code-actual-target-receipt.json', target_receipt)
        working_base = code_directory / 'model.pt'
        original = torch.load(base, map_location='cpu', weights_only=True, mmap=True)
        continued = torch.load(working_base, map_location='cpu', weights_only=True, mmap=True)
        frozen = [name for name in original['model'] if name.startswith('vision.')]
        if (not frozen or original['config'] != continued['config'] or
            any(not torch.equal(original['model'][name], continued['model'][name]) for name in frozen)):
            raise ValueError('Language-only broad-code continuation changed frozen vision/configuration')
        atomic_json(reports / 'broad-code-vision-preservation.json', {'exact_frozen_vision_tensors': len(frozen),
                    'base_checkpoint_sha256': base_sha, 'continued_checkpoint_sha256': file_hash(working_base), 'goal_achieved': False})
        del original, continued
        for snapshot in code_directory.glob('step-*.pt'): snapshot.unlink()
        if (code_directory / 'training-state.pt').exists(): retire(code_directory, reports / 'broad-code-completed-state-cleanup.json')
        systems_root, systems_receipt = admit_systems_variety(ROOT)
        atomic_json(reports / 'systems-variety-input-admission.json', systems_receipt)
        if systems_receipt['tokenizer_sha256'] != broad['tokenizer_sha256']:
            raise ValueError('New systems code and completed own policy tokenizers differ')
        systems_windows = PackedCorpus(systems_root, systems_receipt['tokenizer_sha256'], context)
        systems_steps = math.ceil(len(systems_windows) / 64)
        systems_directory = checkpoints / 'systems-variety-code'
        systems_spec = {'stage': 'pretrain', 'steps': systems_steps, 'trainable': 'language', 'image_size': 448,
                        'data_sha256': systems_receipt['packed_manifest_sha256']}
        systems_base_sha = file_hash(working_base)
        atomic_json(reports / 'systems-variety-code-plan.json', {**systems_spec, 'base_checkpoint_sha256': systems_base_sha,
            'input_admission_sha256': file_hash(reports / 'systems-variety-input-admission.json'),
            'new_public_code_train_tokens': systems_receipt['new_train_tokens'],
            'all_earlier_code_train_tokens_rehearsed': systems_receipt['all_prior_code_train_tokens_rehearsed'],
            'combined_train_tokens': systems_receipt['combined_train_tokens'], 'packed_windows': len(systems_windows),
            'world_size': 4, 'microbatch': 1, 'gradient_accumulation': 16, 'learning_rate': 2e-5,
            'nominal_target_exposures': systems_steps * 64 * (context - 1),
            'packed_tail_token_positions_not_sampled': systems_receipt['combined_train_tokens'] - 1 - len(systems_windows) * (context - 1),
            'final_batch_repeated_windows': systems_steps * 64 - len(systems_windows),
            'scope': 'New original Linux/RTOS/filesystem/network/crypto/security source; complete older code rehearsal, own language weights only.',
            'goal_achieved': False})
        if not complete_stage(systems_directory, systems_spec):
            if systems_directory.exists() and any(systems_directory.iterdir()):
                raise ValueError('Incomplete systems-code continuation needs inspected resume')
            if shutil.disk_usage(ROOT).free < 25 * 1024**3: raise ValueError('Insufficient free disk for systems-code training')
            status('training original public OS/network/security code with full earlier code rehearsal',
                   steps=systems_steps, new_source_tokens=systems_receipt['new_train_tokens'])
            run([str(ROOT / '.venv/bin/torchrun'), '--standalone', '--nproc_per_node=4', '-m', 'slm_lab.training',
                'pretrain', '--checkpoint', str(working_base), '--tokenizer', str(base.parent / 'tokenizer.json'),
                '--data', str(systems_root), '--output', str(systems_directory), '--trainable', 'language',
                '--steps', str(systems_steps), '--batch-size', '1', '--gradient-accumulation', '16', '--learning-rate', '2e-5',
                '--image-size', '448', '--device', 'cuda', '--threads', '1', '--gradient-checkpointing',
                '--checkpoint-every', '250', '--keep-checkpoints', '1', '--log-every', '10'])
            if not complete_stage(systems_directory, systems_spec): raise ValueError('Systems-code continuation did not finish its budget')
        atomic_json(reports / 'systems-variety-actual-target-receipt.json',
                    verify_targets(json.loads((systems_directory / 'training.json').read_text()), systems_steps, context))
        previous_code = torch.load(working_base, map_location='cpu', weights_only=True, mmap=True)
        systems_checkpoint = systems_directory / 'model.pt'
        systems_model = torch.load(systems_checkpoint, map_location='cpu', weights_only=True, mmap=True)
        if (previous_code['config'] != systems_model['config'] or previous_code['tokenizer_sha256'] != systems_model['tokenizer_sha256']
                or any(not torch.equal(previous_code['model'][name], systems_model['model'][name]) for name in frozen)):
            raise ValueError('Systems language learning changed frozen vision/configuration/tokenizer')
        atomic_json(reports / 'systems-variety-vision-preservation.json', {'exact_frozen_vision_tensors': len(frozen),
                    'base_checkpoint_sha256': systems_base_sha, 'candidate_checkpoint_sha256': file_hash(systems_checkpoint), 'goal_achieved': False})
        del previous_code, systems_model
        working_base = systems_checkpoint
        for snapshot in systems_directory.glob('step-*.pt'): snapshot.unlink()
        if (systems_directory / 'training-state.pt').exists(): retire(systems_directory, reports / 'systems-variety-completed-state-cleanup.json')
        while not (clevr / 'manifest.json').exists():
            if clevr_identity is None or process_identity(args.clevr_pid) != clevr_identity:
                raise ValueError('Original CLEVR collector ended without a finalized dataset; inspect before training')
            if capture(files) != loaded: raise ValueError('Retention source changed while waiting for original CLEVR')
            status('waiting for finalized original CC-BY visual reasoning images', clevr_pid=args.clevr_pid)
            time.sleep(15)
        clevr_manifest = json.loads((clevr / 'manifest.json').read_text())
        if clevr_manifest['implementation_sha256'] != file_hash(ROOT / 'src/slm_lab/clevr_data.py'):
            raise ValueError('Original CLEVR preparation implementation changed')
        clevr_audit_path = ROOT / 'reports/clevr-original-input-audit-v1.json'
        if not clevr_audit_path.exists(): raise ValueError('Original CLEVR input audit absent; inspect auditor before visual admission')
        clevr_audit = json.loads(clevr_audit_path.read_text())
        if (clevr_audit.get('passed') is not True or clevr_audit['manifest_sha256'] != file_hash(clevr / 'manifest.json')
                or clevr_audit['audit_source_sha256'] != file_hash(ROOT / 'scripts/audit_clevr_inputs.py')
                or clevr_audit['splits'] != clevr_manifest['splits'] or clevr_audit['files_sha256'] != clevr_manifest['files_sha256']):
            raise ValueError('Original CLEVR input audit differs from finalized data/implementation')
        for relative, digest in clevr_audit['source_files_sha256'].items():
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT) or file_hash(path) != digest:
                raise ValueError('Original CLEVR audited implementation changed before admission')
        atomic_json(reports / 'clevr-input-audit-admission.json', {'passed': True,
                    'audit_sha256': file_hash(clevr_audit_path), 'manifest_sha256': clevr_audit['manifest_sha256'],
                    'scope': 'Original source/pixel/token input audit required before visual training; capability unverified.',
                    'goal_achieved': False})
        for name, digest in clevr_manifest['files_sha256'].items():
            if file_hash(clevr / name) != digest: raise ValueError('Original visual reasoning data changed')
        clevr_rows = list(jsonl(clevr / 'train.jsonl')); clevr_valid = clevr / 'valid.jsonl'
        clevr_valid_rows = list(jsonl(clevr_valid))
        if (len(clevr_rows) != clevr_manifest['splits']['train']['examples'] or
            len(clevr_valid_rows) != clevr_manifest['splits']['valid']['examples'] or
            {row['group_id'] for row in clevr_rows} & {row['group_id'] for row in clevr_valid_rows}):
            raise ValueError('Original CLEVR counts/image split differ')
        checked = set()
        for row in clevr_rows + clevr_valid_rows:
            path = (clevr / row['image']).resolve()
            if not path.is_relative_to(clevr.resolve()): raise ValueError('Visual reasoning image path escapes its own dataset')
            if path not in checked:
                if file_hash(path) != row['stored_image_sha256']: raise ValueError('Original visual reasoning image changed')
                checked.add(path)
        clevr_rows = [{**row, 'image': str((clevr / row['image']).resolve()),
                       'curriculum_source': 'original_clevr_program_verified'} for row in clevr_rows]
        photos = ROOT / 'data/expanded-everyday-photos-v1'
        while not (photos / 'manifest.json').exists():
            if photo_identity is None or process_identity(args.photos_pid) != photo_identity:
                raise ValueError('Additional-photo collector ended without a finalized dataset; inspect before training')
            if capture(files) != loaded:
                raise ValueError('Retention source/runtime changed while waiting for finalized photos')
            status('waiting for finalized additional photos after completed main curriculum', photos_pid=args.photos_pid,
                   photos_start_ticks=photo_identity)
            time.sleep(15)
        photo_manifest = json.loads((photos / 'manifest.json').read_text())
        if photo_manifest['configuration']['implementation_sha256'] != file_hash(ROOT / 'src/slm_lab/expanded_photos.py'):
            raise ValueError('Additional-photo preparation implementation differs')
        for name, digest in photo_manifest['files_sha256'].items():
            if file_hash(photos / name) != digest:
                raise ValueError('Finalized additional-photo records changed')
        photo_rows = list(jsonl(photos / 'train.jsonl'))
        if len(photo_rows) != photo_manifest['train_examples'] or not photo_rows:
            raise ValueError('Finalized additional-photo image count differs')
        for row in photo_rows:
            path = (photos / row['image']).resolve()
            if not path.is_relative_to(photos.resolve()) or file_hash(path) != row['stored_image_sha256']:
                raise ValueError('Finalized additional-photo image/path changed')
            row['image'] = str(path)
        while not (words / 'manifest.json').exists():
            if words_identity is None or process_identity(args.words_pid) != words_identity:
                raise ValueError('Scene-word preparation ended without a finalized dataset')
            if capture(files) != loaded: raise ValueError('Retention source/runtime changed while waiting for word data')
            status('waiting for finalized additional real-word crops', words_pid=args.words_pid)
            time.sleep(15)
        word_manifest = json.loads((words / 'manifest.json').read_text())
        if word_manifest['configuration']['implementation_sha256'] != file_hash(ROOT / 'src/slm_lab/expanded_scene_words.py'):
            raise ValueError('Additional scene-word preparation changed')
        for name, digest in word_manifest['files_sha256'].items():
            if file_hash(words / name) != digest: raise ValueError('Additional scene-word records changed')
        word_rows = list(jsonl(words / 'train.jsonl'))
        if len(word_rows) != word_manifest['train_examples'] or not word_rows:
            raise ValueError('Additional real-word count differs')
        for row in word_rows:
            path = Path(row['image']).resolve()
            if not path.is_relative_to(words.resolve()) or file_hash(path) != row['stored_image_sha256']:
                raise ValueError('Additional scene-word image/path changed')
        real_reading_rows, real_reading_receipt = admit_real_reading(ROOT)
        atomic_json(reports / 'textvqa-input-audit-admission.json', real_reading_receipt)
        real_reading_data = ROOT / 'data/textvqa-human-reading-v1'
        own_context_rows, own_context_receipt = admit_own_context_reading(ROOT)
        atomic_json(reports / 'own-context-input-audit-admission.json', own_context_receipt)
        own_context_data = ROOT / 'data/own-ocr-human-questions-v1'
        human_fact_rows, human_fact_receipt = admit_human_fact_questions(ROOT)
        atomic_json(reports / 'human-fact-question-input-admission.json', human_fact_receipt)
        human_fact_data = ROOT / 'data/nq-human-questions-v1'
        complete_js_rows, complete_js_receipt = admit_complete_js_parameters(ROOT)
        atomic_json(reports / 'complete-js-parameter-input-admission.json', complete_js_receipt)
        complete_js_data = ROOT / 'data/complete-js-parameters-v1'
        goruby_rows, goruby_receipt = admit_goruby_headers(ROOT)
        atomic_json(reports / 'goruby-header-input-admission.json', goruby_receipt)
        goruby_data = ROOT / 'data/complete-goruby-headers-v1'
        while not (paired_photo_data / 'manifest.json').exists():
            if paired_photo_identity is None or process_identity(args.vqa_pairs_pid) != paired_photo_identity:
                raise ValueError('Original paired-photo collector ended without finalization; inspect before admission')
            if capture(files) != loaded: raise ValueError('Retention source changed while waiting for paired photos')
            status('waiting for finalized original paired real-photo questions',
                   vqa_pairs_pid=args.vqa_pairs_pid, vqa_pairs_start_ticks=paired_photo_identity)
            time.sleep(15)
        paired_photo_rows, paired_photo_receipt = admit_paired_photos(ROOT)
        atomic_json(reports / 'vqa-pair-input-admission.json', paired_photo_receipt)
        exercises = ROOT / 'data/exercism-code-v1'
        exercise_manifest = json.loads((exercises / 'manifest.json').read_text())
        if exercise_manifest['implementation_sha256'] != file_hash(ROOT / 'src/slm_lab/exercise_code.py'):
            raise ValueError('Exercise source preparation implementation changed')
        for name, digest in exercise_manifest['files_sha256'].items():
            if file_hash(exercises / name) != digest: raise ValueError('Finalized programming exercises changed')
        exercise_rows, exercise_support_receipt = admit_exercise_support_context(ROOT)
        atomic_json(reports / 'exercise-support-context-input-admission.json', exercise_support_receipt)
        exercise_valid_rows = list(jsonl(exercises / 'valid.jsonl'))
        if (len(exercise_rows) != exercise_support_receipt['examples']
                or len(exercise_valid_rows) != exercise_manifest['splits']['valid']['examples']
                or {r['group_id'] for r in exercise_rows} & {r['group_id'] for r in exercise_valid_rows}
                or any(r['split'] != 'train' or r.get('image') for r in exercise_rows)):
            raise ValueError('Exercise counts or cross-language parent splits differ')
        for row in exercise_rows + exercise_valid_rows:
            for item in row['published_test_artifacts']:
                path = (exercises / item['artifact']).resolve()
                if not path.is_relative_to(exercises.resolve()) or file_hash(path) != item['sha256']:
                    raise ValueError('Published exercise test artifact changed')
        # Preserve publisher algorithms, declared adaptations and attribution. Repetition is explicitly
        # an exposure budget, not additional distinct source programs.
        exercise_repetitions = 4
        repeated_exercises = [{**row, 'id': row['id'] + '-replay-' + str(index),
                               'original_exercise_record_id': row['id'], 'exercise_replay_index': index,
                               'curriculum_source': 'original_exercism_tasks'}
                              for index in range(exercise_repetitions) for row in exercise_rows]
        dialogue = ROOT / 'data/human-dialogue-v1'
        dialogue_manifest = json.loads((dialogue / 'manifest.json').read_text())
        for name, digest in dialogue_manifest['files_sha256'].items():
            if file_hash(dialogue / name) != digest:
                raise ValueError('Reviewed human dialogue records changed')
        curriculum = ROOT / 'data/extended-curriculum-v1'
        preceding_plan = json.loads((preceding_reports / 'plan.json').read_text())
        joint = next(stage for stage in preceding_plan['stages'] if stage['name'] == 'joint')
        if file_hash(curriculum / 'joint.train.jsonl') != joint['data_sha256']:
            raise ValueError('Previous code/math/instruction/image rehearsal source changed')
        visual = ROOT / 'data/visual-pretraining-v1'
        visual_manifest = json.loads((visual / 'manifest.json').read_text())
        if file_hash(visual / 'train.jsonl') != visual_manifest['files_sha256']['train.jsonl']:
            raise ValueError('Completed Commons/photo supplement changed')
        data = ROOT / 'data/dialogue-retention-v1'; data.mkdir(exist_ok=True)
        destination = data / 'joint.train.jsonl'
        existing_rows = list(jsonl(curriculum / 'joint.train.jsonl'))
        seen_groups = {row['group_id'] for row in existing_rows if row.get('image')}
        extra_images = [row for row in jsonl(visual / 'train.jsonl')
                        if row['visual_source'] == 'commons' and row['group_id'] not in seen_groups]
        dialogues = list(jsonl(dialogue / 'train.jsonl'))
        # Preserve every prior rehearsal record and its source metadata. Source
        # responses remain original; the human-only supplement is never used as
        # a substitute for verified code/math or grounded image annotations.
        if any(row['group_id'] in seen_groups for row in photo_rows):
            raise ValueError('Additional photo parents overlap previous joint training')
        if any(row['group_id'] not in seen_groups or row['task'] != 'word_ocr' for row in word_rows):
            raise ValueError('Additional words require existing admitted training parents')
        if {row['id'] for row in word_rows} & {row['id'] for row in existing_rows}:
            raise ValueError('Repeated original/additional word record')
        admitted_photo_groups = seen_groups | {row['group_id'] for row in photo_rows}
        if any(row['group_id'] not in admitted_photo_groups for row in real_reading_rows):
            raise ValueError('Original reading questions require already admitted full training photos')
        if {row['id'] for row in real_reading_rows} & {row['id'] for row in existing_rows}:
            raise ValueError('Repeated original human reading question record')
        if (len(own_context_rows) != len(real_reading_rows)
                or {row['source_question_id'] for row in own_context_rows} != {row['id'] for row in real_reading_rows}
                or any(row['group_id'] not in admitted_photo_groups or row['split'] != 'train'
                       or row['own_predicted_context_supplied'] is not True for row in own_context_rows)):
            raise ValueError('Own-context rows must preserve every original training question and admitted full photo')
        prior_rows = existing_rows + dialogues + extra_images + photo_rows + word_rows + repeated_exercises + clevr_rows + real_reading_rows
        if {row['id'] for row in own_context_rows} & {row['id'] for row in prior_rows}:
            raise ValueError('Repeated own-context human question variant')
        rows = prior_rows + own_context_rows
        if {row['id'] for row in human_fact_rows} & {row['id'] for row in rows}:
            raise ValueError('Repeated original human fact question record')
        if len(human_fact_rows) != 87925 or any(row['split'] != 'train' or row.get('image') for row in human_fact_rows):
            raise ValueError('Human fact rehearsal must admit only all original training questions')
        rows += human_fact_rows
        if (len(complete_js_rows) != 982 or any(row['split'] != 'train' or row.get('image') for row in complete_js_rows)
                or {row['id'] for row in complete_js_rows} & {row['id'] for row in rows}):
            raise ValueError('Complete JavaScript parameter variants must be unique admitted training-only rows')
        rows += complete_js_rows
        if (len(goruby_rows) != 47006 or any(row['split'] != 'train' or row.get('image') for row in goruby_rows)
                or {row['id'] for row in goruby_rows} & {row['id'] for row in rows}):
            raise ValueError('Complete Go/Ruby headers must be unique admitted training-only variants')
        rows += goruby_rows
        if (not paired_photo_rows or any(row['split'] != 'train' or not row.get('image') for row in paired_photo_rows)
                or {row['id'] for row in paired_photo_rows} & {row['id'] for row in rows}):
            raise ValueError('Original paired-photo questions must be unique finalized training-only rows')
        rows += paired_photo_rows
        if destination.exists():
            expected = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
            if destination.read_text() != expected:
                raise ValueError('Prepared dialogue rehearsal differs')
        else:
            destination.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        source = file_hash(destination); steps = math.ceil(len(rows) / 16)
        modality_counts = {name: sum(bool(row.get('image')) == flag for row in rows)
                           for name, flag in [('text', False), ('image', True)]}
        plan = {'base_checkpoint': str(base), 'base_checkpoint_sha256': base_sha,
                'dialogue_manifest_sha256': file_hash(dialogue / 'manifest.json'),
                'previous_joint_sha256': joint['data_sha256'], 'visual_manifest_sha256': file_hash(visual / 'manifest.json'),
                'data_sha256': source, 'source_records': len(rows), 'human_dialogues': len(dialogues),
                'rehearsal_records': len(existing_rows), 'additional_commons_captions': len(extra_images),
                'additional_human_captioned_photos': len(photo_rows),
                'additional_photo_manifest_sha256': file_hash(photos / 'manifest.json'),
                'original_clevr_question_records': len(clevr_rows),
                'clevr_manifest_sha256': file_hash(clevr / 'manifest.json'),
                'original_human_reading_question_records': len(real_reading_rows),
                'textvqa_manifest_sha256': file_hash(real_reading_data / 'manifest.json'),
                'textvqa_input_audit_sha256': real_reading_receipt['audit_sha256'],
                'own_predicted_context_question_variants': len(own_context_rows),
                'own_context_manifest_sha256': file_hash(own_context_data / 'manifest.json'),
                'own_context_admission_receipt_sha256': file_hash(reports / 'own-context-input-audit-admission.json'),
                'plain_human_questions_preserved': True,
                'original_human_fact_question_records': len(human_fact_rows),
                'human_fact_question_manifest_sha256': file_hash(human_fact_data / 'manifest.json'),
                'human_fact_input_audit_sha256': human_fact_receipt['input_audit_sha256'],
                'human_fact_exposure_audit_sha256': human_fact_receipt['exposure_audit_sha256'],
                'human_fact_development_questions_with_exact_potential_exposure': len(human_fact_receipt['development_cases_with_exact_potential_training_exposure']),
                'complete_original_JS_parameter_instruction_variants': len(complete_js_rows),
                'complete_JS_parameter_manifest_sha256': file_hash(complete_js_data / 'manifest.json'),
                'complete_JS_parameter_targets_per_source_pass': complete_js_receipt['actual_targets_per_pass'],
                'complete_JS_parameter_admission_receipt_sha256': file_hash(reports / 'complete-js-parameter-input-admission.json'),
                'complete_original_Go_Ruby_header_variants': len(goruby_rows),
                'complete_Go_Ruby_header_manifest_sha256': file_hash(goruby_data / 'manifest.json'),
                'complete_Go_Ruby_header_targets_per_source_pass': goruby_receipt['actual_targets_per_source_pass'],
                'complete_Go_Ruby_header_admission_receipt_sha256': file_hash(reports / 'goruby-header-input-admission.json'),
                'original_paired_real_photo_question_records': len(paired_photo_rows),
                'original_paired_real_photo_manifest_sha256': paired_photo_receipt['manifest_sha256'],
                'original_paired_real_photo_targets_per_source_pass': paired_photo_receipt['splits']['train']['actual_target_tokens_per_source_pass'],
                'paired_photo_admission_receipt_sha256': file_hash(reports / 'vqa-pair-input-admission.json'),
                'additional_real_word_crops': len(word_rows), 'word_manifest_sha256': file_hash(words / 'manifest.json'),
                'additional_public_exercise_implementations': len(exercise_rows),
                'additional_exercise_record_exposures': len(repeated_exercises),
                'exercise_repetitions': exercise_repetitions, 'exercise_manifest_sha256': file_hash(exercises / 'manifest.json'),
                'exercise_support_context_manifest_sha256': exercise_support_receipt['manifest_sha256'],
                'exercise_support_context_admission_sha256': file_hash(reports / 'exercise-support-context-input-admission.json'),
                'modality_counts': modality_counts, 'sampler_incomplete_global_modality_batch_rows': sum(value % 8 for value in modality_counts.values()),
                'passes': 1, 'steps': steps, 'world_size': 4, 'microbatch': 2, 'accumulation': 2,
                'learning_rate': 1e-5, 'trainable': 'all', 'image_size': 448,
                'selection_guard_sha256': file_hash(ROOT / 'src/slm_lab/alignment_selection.py'),
                'loaded_source_runtime': loaded, 'goal_achieved': False,
                'scope': 'Dedicated real-word vision learning then joint captioned photos/original paired-photo questions/plain and own-context reading/original human fact questions/dialogues/full prior rehearsal; own model only.',
                'limitations': ['Inferred human quality is not independent correctness; sparse languages remain weakly represented.',
                                'Own-context variants repeat original human targets with noisy own predictions; they are not new independent questions.',
                                'NQ exact exposure findings remain explicit; complete original development scores do not certify semantic or article independence.',
                                'Homogeneous global minibatches omit each modality remainder below eight rows; repeats fill the finite step budget.',
                                'Full local capability/offline preference guards are tuning criteria, not high-end capability or own-policy human ratings.']}
        path = reports / 'plan.json'
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError('Dialogue-retention plan differs from previous preparation')
        atomic_json(path, plan)
        reading_rows = [row for row in existing_rows if row.get('task') == 'word_ocr'] + word_rows
        reading_data = data / 'real-word.train.jsonl'
        reading_data.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in reading_rows))
        reading_steps = math.ceil(len(reading_rows) / 16)
        reading_directory = checkpoints / 'reading'
        reading_spec = {'stage': 'sft', 'steps': reading_steps, 'trainable': 'vision', 'image_size': 448,
                        'data_sha256': file_hash(reading_data)}
        atomic_json(reports / 'reading-plan.json', {**reading_spec, 'examples': len(reading_rows),
                    'new_word_examples': len(word_rows), 'passes': 1, 'learning_rate': 2e-5,
                    'world_size': 4, 'microbatch': 2, 'accumulation': 2, 'base_checkpoint_sha256': file_hash(working_base),
                    'scope': 'Dedicated real-word response targets; language/critic frozen; native reading not established.',
                    'goal_achieved': False})
        if not complete_stage(reading_directory, reading_spec):
            if reading_directory.exists() and any(reading_directory.iterdir()):
                raise ValueError('Incomplete dedicated reading stage requires inspected resume')
            if shutil.disk_usage(ROOT).free < 25 * 1024**3:
                raise ValueError('Insufficient disk for reading warmup')
            status('training original and additional real words with frozen language', steps=reading_steps, examples=len(reading_rows))
            run([str(ROOT / '.venv/bin/torchrun'), '--standalone', '--nproc_per_node=4', '-m', 'slm_lab.training', 'sft',
                 '--checkpoint', str(working_base), '--tokenizer', str(base.parent / 'tokenizer.json'), '--data', str(reading_data),
                 '--output', str(reading_directory), '--trainable', 'vision', '--steps', str(reading_steps),
                 '--batch-size', '2', '--gradient-accumulation', '2', '--learning-rate', '2e-5', '--image-size', '448',
                 '--device', 'cuda', '--threads', '1', '--gradient-checkpointing', '--checkpoint-every', '1000',
                 '--keep-checkpoints', '1', '--log-every', '10'])
            if not complete_stage(reading_directory, reading_spec): raise ValueError('Reading warmup did not complete')
        reading_checkpoint = reading_directory / 'model.pt'
        before = torch.load(working_base, map_location='cpu', weights_only=True, mmap=True)
        after = torch.load(reading_checkpoint, map_location='cpu', weights_only=True, mmap=True)
        language_sha = language_digest(before['model'])
        if language_digest(after['model']) != language_sha: raise ValueError('Real-word warmup changed frozen language/critic')
        atomic_json(reports / 'reading-language-preservation.json', {'language_state_sha256': language_sha,
                    'exact_language_and_critic_preserved': True, 'goal_achieved': False})
        del before, after
        for snapshot in reading_directory.glob('step-*.pt'): snapshot.unlink()
        if (reading_directory / 'training-state.pt').exists():
            retire(reading_directory, reports / 'reading-completed-state-cleanup.json')
        dialogue_valid = dialogue / 'valid.jsonl'
        valid_count = dialogue_manifest['splits']['valid']['examples']
        exercise_valid = exercises / 'valid.jsonl'
        cpp_references = reports / 'exercise-cpp-references.json'
        status('verifying all original held-out C++ publisher references before policy assessments')
        run([sys.executable, '-u', '-m', 'slm_lab.exercise_cpp_evaluation', '--verify-references',
             '--data', str(exercise_valid), '--output', str(cpp_references)])
        js_references = json.loads((ROOT / 'reports/exercism-javascript-publisher-references-v1.json').read_text())
        if (js_references['complete_all37_reference_suites_passed'] is not True
                or js_references['examples'] != 37 or js_references['passed'] != 37
                or js_references['data_sha256'] != file_hash(exercise_valid)):
            raise ValueError('All original held JavaScript references must pass before policy assessments')
        go_references = json.loads((ROOT / 'reports/exercism-go-publisher-references-v1.json').read_text())
        if (go_references['complete_all42_reference_suites_passed'] is not True
                or go_references['examples'] != 42 or go_references['passed'] != 42
                or go_references['data_sha256'] != file_hash(exercise_valid)):
            raise ValueError('All original held Go references must pass before policy assessments')
        rust_references = json.loads((ROOT / 'reports/exercism-rust-publisher-references-v2.json').read_text())
        if (rust_references['complete_all24_reference_suites_passed'] is not True
                or rust_references['examples'] != 24 or rust_references['passed'] != 24
                or rust_references['data_sha256'] != file_hash(exercise_valid)):
            raise ValueError('All original held Rust references must pass before policy assessments')
        for module, output, extra in [
            ('slm_lab.evaluation', 'exercise-loss-before.json', ['--task', 'loss', '--max-examples', str(len(exercise_valid_rows)), '--threads', '1']),
            ('slm_lab.exercise_evaluation', 'exercise-code-before.json', []),
            ('slm_lab.exercise_cpp_evaluation', 'exercise-cpp-before.json', ['--reference-report', str(cpp_references)]),
            ('slm_lab.exercise_js_evaluation', 'exercise-js-before.json', []),
            ('slm_lab.exercise_go_evaluation', 'exercise-go-before.json', []),
            ('slm_lab.exercise_rust_evaluation_v2', 'exercise-rust-before.json', [])]:
            status('assessing original held-out exercise interfaces before joint training', assessment=output)
            run([sys.executable, '-u', '-m', module, '--checkpoint', str(base), '--data', str(exercise_valid),
                 '--device', 'cuda', '--output', str(reports / output), *extra])
        status('assessing original held visual reasoning before joint training')
        run([sys.executable, '-u', '-m', 'slm_lab.clevr_evaluation', '--checkpoint', str(base), '--data', str(clevr_valid),
             '--output', str(reports / 'clevr-before.json'), '--device', 'cuda'])
        status('assessing original held-out real-photo reading questions before joint training')
        run([sys.executable, '-u', '-m', 'slm_lab.textvqa_evaluation', '--checkpoint', str(base),
             '--data', str(real_reading_data), '--output', str(reports / 'textvqa-before.json'), '--device', 'cuda'])
        status('assessing compact own-reader native answers before joint training')
        run([sys.executable, '-u', '-m', 'slm_lab.own_context_transfer_evaluation', '--checkpoint', str(base),
             '--output', str(reports / 'own-context-before.json'), '--device', 'cuda'])
        status('assessing all original human fact questions before joint training')
        run([sys.executable, '-u', '-m', 'slm_lab.nq_open_evaluation', '--checkpoint', str(base),
             '--output', str(reports / 'nq-open-before.json'), '--device', 'cuda'])
        status('assessing original paired photos with correct/blank/counterpart controls before joint training')
        run([sys.executable, '-u', '-m', 'slm_lab.vqa_pair_evaluation', '--checkpoint', str(base),
             '--data', str(paired_photo_data), '--output', str(reports / 'vqa-pairs-before.json'), '--device', 'cuda'])
        atomic_json(reports / 'vqa-pair-assessment-coverage.json',
                    validate_paired_photos(json.loads((reports / 'vqa-pairs-before.json').read_text()), paired_photo_data))
        status('measuring full held-out human-dialogue response loss before retention training')
        run([sys.executable, '-u', '-m', 'slm_lab.evaluation', '--task', 'loss', '--checkpoint', str(base),
             '--data', str(dialogue_valid), '--max-examples', str(valid_count), '--threads', '1', '--device', 'cuda',
             '--output', str(reports / 'dialogue-before.json')])
        directory = checkpoints / 'joint'
        spec = {'stage': 'sft', 'steps': steps, 'trainable': 'all', 'image_size': 448, 'data_sha256': source}
        if not complete_stage(directory, spec):
            if directory.exists() and any(directory.iterdir()):
                raise ValueError('Incomplete dialogue-retention state needs inspected resume')
            if shutil.disk_usage(ROOT).free < 25 * 1024**3:
                raise ValueError('Insufficient free disk for new dialogue-retention checkpoints')
            status('joint training additional photos and reviewed dialogues with full rehearsal', steps=steps, source_records=len(rows),
                   additional_photos=len(photo_rows))
            run([str(ROOT / '.venv/bin/torchrun'), '--standalone', '--nproc_per_node=4', '-m', 'slm_lab.training', 'sft',
                 '--checkpoint', str(reading_checkpoint), '--tokenizer', str(base.parent / 'tokenizer.json'), '--data', str(destination),
                 '--output', str(directory), '--trainable', 'all', '--steps', str(steps), '--batch-size', '2',
                 '--gradient-accumulation', '2', '--learning-rate', '1e-5', '--image-size', '448', '--device', 'cuda',
                 '--threads', '1', '--gradient-checkpointing', '--checkpoint-every', '1000', '--keep-checkpoints', '1', '--log-every', '10'])
            if not complete_stage(directory, spec):
                raise ValueError('Dialogue-retention training did not finish')
        for snapshot in directory.glob('step-*.pt'): snapshot.unlink()
        candidate = directory / 'model.pt'
        for module, output, extra in [
            ('slm_lab.evaluation', 'exercise-loss-after.json', ['--task', 'loss', '--max-examples', str(len(exercise_valid_rows)), '--threads', '1']),
            ('slm_lab.exercise_evaluation', 'exercise-code-after.json', []),
            ('slm_lab.exercise_cpp_evaluation', 'exercise-cpp-after.json', ['--reference-report', str(cpp_references)]),
            ('slm_lab.exercise_js_evaluation', 'exercise-js-after.json', []),
            ('slm_lab.exercise_go_evaluation', 'exercise-go-after.json', []),
            ('slm_lab.exercise_rust_evaluation_v2', 'exercise-rust-after.json', [])]:
            status('assessing original held-out exercise interfaces after joint training', assessment=output)
            run([sys.executable, '-u', '-m', module, '--checkpoint', str(candidate), '--data', str(exercise_valid),
                 '--device', 'cuda', '--output', str(reports / output), *extra])
        status('assessing original held visual reasoning after joint training')
        run([sys.executable, '-u', '-m', 'slm_lab.clevr_evaluation', '--checkpoint', str(candidate), '--data', str(clevr_valid),
             '--output', str(reports / 'clevr-after.json'), '--device', 'cuda'])
        status('assessing original held-out real-photo reading questions after joint training')
        run([sys.executable, '-u', '-m', 'slm_lab.textvqa_evaluation', '--checkpoint', str(candidate),
             '--data', str(real_reading_data), '--output', str(reports / 'textvqa-after.json'), '--device', 'cuda'])
        status('assessing compact own-reader native answers after joint training')
        run([sys.executable, '-u', '-m', 'slm_lab.own_context_transfer_evaluation', '--checkpoint', str(candidate),
             '--output', str(reports / 'own-context-after.json'), '--device', 'cuda'])
        status('assessing all original human fact questions after joint training')
        run([sys.executable, '-u', '-m', 'slm_lab.nq_open_evaluation', '--checkpoint', str(candidate),
             '--output', str(reports / 'nq-open-after.json'), '--device', 'cuda'])
        status('assessing original paired photos with correct/blank/counterpart controls after joint training')
        run([sys.executable, '-u', '-m', 'slm_lab.vqa_pair_evaluation', '--checkpoint', str(candidate),
             '--data', str(paired_photo_data), '--output', str(reports / 'vqa-pairs-after.json'), '--device', 'cuda'])
        status('measuring full held-out human-dialogue response loss after retention training')
        run([sys.executable, '-u', '-m', 'slm_lab.evaluation', '--task', 'loss', '--checkpoint', str(candidate),
             '--data', str(dialogue_valid), '--max-examples', str(valid_count), '--threads', '1', '--device', 'cuda',
             '--output', str(reports / 'dialogue-after.json')])
        for family, module, name, extra in specifications(ROOT):
            status('evaluating completed dialogue-retention policy', task=module)
            run([sys.executable, '-u', '-m', module, '--checkpoint', str(candidate), '--output', str(reports / name),
                 '--device', 'cuda', *extra])
        status('measuring held-out human-pair likelihood relative to the preceding policy')
        run([sys.executable, '-u', '-m', 'slm_lab.evaluation', '--task', 'policy_preference', '--checkpoint', str(candidate),
             '--reference-checkpoint', str(base), '--data', str(ROOT / 'data/human-preference-curriculum-v1/preferences.valid.jsonl'),
             '--max-examples', '1000', '--threads', '1', '--device', 'cuda', '--output', str(reports / 'policy-preference.json')])
        assessed = capabilities(reports)
        selection = compare(baseline, assessed, json.loads((reports / 'policy-preference.json').read_text()))
        exercise_before = json.loads((reports / 'exercise-code-before.json').read_text())
        exercise_after = json.loads((reports / 'exercise-code-after.json').read_text())
        exercise_guard = compare_exercises(exercise_before, exercise_after)
        if exercise_guard['base_checkpoint_sha256'] != base_sha or exercise_guard['candidate_checkpoint_sha256'] != file_hash(candidate):
            raise ValueError('Exercise assessment model identities differ')
        atomic_json(reports / 'exercise-capability-guard.json', exercise_guard)
        selection['exercise_module_guard'] = exercise_guard
        selection['eligible_for_selection'] = selection['eligible_for_selection'] and exercise_guard['passed']
        cpp_guard = compare_cpp_exercises(json.loads((reports / 'exercise-cpp-before.json').read_text()),
                                         json.loads((reports / 'exercise-cpp-after.json').read_text()))
        if cpp_guard['base_checkpoint_sha256'] != base_sha or cpp_guard['candidate_checkpoint_sha256'] != file_hash(candidate):
            raise ValueError('C++ exercise assessment model identities differ')
        atomic_json(reports / 'exercise-cpp-capability-guard.json', cpp_guard)
        selection['exercise_cpp_module_guard'] = cpp_guard
        selection['eligible_for_selection'] = selection['eligible_for_selection'] and cpp_guard['passed']
        js_guard = compare_js_exercises(json.loads((reports / 'exercise-js-before.json').read_text()),
                                       json.loads((reports / 'exercise-js-after.json').read_text()))
        if js_guard['base_checkpoint_sha256'] != base_sha or js_guard['candidate_checkpoint_sha256'] != file_hash(candidate):
            raise ValueError('JavaScript exercise assessment model identities differ')
        atomic_json(reports / 'exercise-js-capability-guard.json', js_guard)
        selection['exercise_js_module_guard'] = js_guard
        selection['eligible_for_selection'] = selection['eligible_for_selection'] and js_guard['passed']
        go_guard = compare_go_exercises(json.loads((reports / 'exercise-go-before.json').read_text()),
                                       json.loads((reports / 'exercise-go-after.json').read_text()))
        if go_guard['base_checkpoint_sha256'] != base_sha or go_guard['candidate_checkpoint_sha256'] != file_hash(candidate):
            raise ValueError('Go exercise assessment model identities differ')
        atomic_json(reports / 'exercise-go-capability-guard.json', go_guard)
        selection['exercise_go_module_guard'] = go_guard
        selection['eligible_for_selection'] = selection['eligible_for_selection'] and go_guard['passed']
        rust_before = json.loads((reports / 'exercise-rust-before.json').read_text())
        rust_after = json.loads((reports / 'exercise-rust-after.json').read_text())
        validate_rust_coverage(rust_before, rust_references['results'])
        validate_rust_coverage(rust_after, rust_references['results'])
        rust_guard = compare_rust_exercises(rust_before, rust_after)
        if rust_guard['base_checkpoint_sha256'] != base_sha or rust_guard['candidate_checkpoint_sha256'] != file_hash(candidate):
            raise ValueError('Rust exercise assessment model identities differ')
        atomic_json(reports / 'exercise-rust-capability-guard.json', rust_guard)
        selection['exercise_rust_module_guard'] = rust_guard
        selection['eligible_for_selection'] = selection['eligible_for_selection'] and rust_guard['passed']
        visual_guard = compare_visual_reasoning(json.loads((reports / 'clevr-before.json').read_text()),
                                               json.loads((reports / 'clevr-after.json').read_text()))
        if visual_guard['base_checkpoint_sha256'] != base_sha or visual_guard['candidate_checkpoint_sha256'] != file_hash(candidate):
            raise ValueError('Visual reasoning assessment model identities differ')
        atomic_json(reports / 'clevr-capability-guard.json', visual_guard)
        selection['visual_reasoning_guard'] = visual_guard
        selection['eligible_for_selection'] = selection['eligible_for_selection'] and visual_guard['passed']
        real_reading_guard = compare_real_reading(json.loads((reports / 'textvqa-before.json').read_text()),
                                                  json.loads((reports / 'textvqa-after.json').read_text()))
        if real_reading_guard['base_checkpoint_sha256'] != base_sha or real_reading_guard['candidate_checkpoint_sha256'] != file_hash(candidate):
            raise ValueError('Original real-photo reading assessment model identities differ')
        atomic_json(reports / 'textvqa-capability-guard.json', real_reading_guard)
        selection['real_photo_reading_guard'] = real_reading_guard
        selection['eligible_for_selection'] = selection['eligible_for_selection'] and real_reading_guard['passed']
        own_context_guard = compare_own_context_reading(json.loads((reports / 'own-context-before.json').read_text()),
                                                      json.loads((reports / 'own-context-after.json').read_text()))
        if own_context_guard['base_checkpoint_sha256'] != base_sha or own_context_guard['candidate_checkpoint_sha256'] != file_hash(candidate):
            raise ValueError('Own-context native assessment model identities differ')
        atomic_json(reports / 'own-context-capability-guard.json', own_context_guard)
        selection['own_context_reading_guard'] = own_context_guard
        selection['eligible_for_selection'] = selection['eligible_for_selection'] and own_context_guard['passed']
        human_fact_guard = compare_human_fact_questions(json.loads((reports / 'nq-open-before.json').read_text()),
                                                       json.loads((reports / 'nq-open-after.json').read_text()))
        if human_fact_guard['base_checkpoint_sha256'] != base_sha or human_fact_guard['candidate_checkpoint_sha256'] != file_hash(candidate):
            raise ValueError('Original human fact assessment model identities differ')
        atomic_json(reports / 'human-fact-question-capability-guard.json', human_fact_guard)
        selection['human_fact_question_guard'] = human_fact_guard
        selection['eligible_for_selection'] = selection['eligible_for_selection'] and human_fact_guard['passed']
        paired_photo_guard = compare_paired_photos(json.loads((reports / 'vqa-pairs-before.json').read_text()),
                                                  json.loads((reports / 'vqa-pairs-after.json').read_text()), paired_photo_data)
        if paired_photo_guard['base_checkpoint_sha256'] != base_sha or paired_photo_guard['candidate_checkpoint_sha256'] != file_hash(candidate):
            raise ValueError('Original paired-photo assessment model identities differ')
        atomic_json(reports / 'vqa-pair-capability-guard.json', paired_photo_guard)
        selection['paired_photo_guard'] = paired_photo_guard
        selection['eligible_for_selection'] = selection['eligible_for_selection'] and paired_photo_guard['passed']
        if selection['base_checkpoint_sha256'] != base_sha or selection['candidate_checkpoint_sha256'] != file_hash(candidate):
            raise ValueError('Dialogue policy/selection evidence differs')
        chosen = candidate if selection['eligible_for_selection'] else base
        source_reports = reports if selection['eligible_for_selection'] else preceding_reports
        selected = checkpoints / 'selected'; selected.mkdir(exist_ok=True)
        alias = selected / 'model.pt'
        if alias.exists():
            if file_hash(alias) != file_hash(chosen): raise ValueError('Selected policy alias differs')
        else:
            os.link(chosen, alias)
        shutil.copy2(chosen.parent / 'tokenizer.json', selected / 'tokenizer.json')
        selected_reports = reports / 'selected'; selected_reports.mkdir(exist_ok=True)
        for name in REPORT_FILES.values(): shutil.copy2(source_reports / name, selected_reports / name)
        shutil.copy2(reports / ('textvqa-after.json' if selection['eligible_for_selection'] else 'textvqa-before.json'),
                     selected_reports / 'textvqa.json')
        shutil.copy2(reports / ('own-context-after.json' if selection['eligible_for_selection'] else 'own-context-before.json'),
                     selected_reports / 'own-context.json')
        shutil.copy2(reports / ('nq-open-after.json' if selection['eligible_for_selection'] else 'nq-open-before.json'),
                     selected_reports / 'nq-open.json')
        shutil.copy2(reports / ('exercise-js-after.json' if selection['eligible_for_selection'] else 'exercise-js-before.json'),
                     selected_reports / 'exercise-js.json')
        shutil.copy2(reports / ('exercise-go-after.json' if selection['eligible_for_selection'] else 'exercise-go-before.json'),
                     selected_reports / 'exercise-go.json')
        shutil.copy2(reports / ('exercise-rust-after.json' if selection['eligible_for_selection'] else 'exercise-rust-before.json'),
                     selected_reports / 'exercise-rust.json')
        shutil.copy2(reports / ('vqa-pairs-after.json' if selection['eligible_for_selection'] else 'vqa-pairs-before.json'),
                     selected_reports / 'vqa-pairs.json')
        atomic_json(reports / 'selection.json', {**selection, 'selected_checkpoint': str(alias),
                    'selected_checkpoint_sha256': file_hash(alias), 'candidate_selected': selection['eligible_for_selection'],
                    'selection_rule': 'All nine declared capability guards, retained Python/C++/JavaScript/Go/Rust exercise modules, original visual reasoning and paired real-photo answers, plain and compact own-reader native answers, all original human fact questions, and positive offline human-pair improvement; retain preceding base otherwise.',
                    'goal_achieved': False})
        if (directory / 'training-state.pt').exists():
            # All candidate assessments and the selected alias are finalized.
            # The next feedback stage reads retained weights, not this optimizer.
            retire(directory, reports / 'joint-completed-state-cleanup.json')
        status(DONE, selected_checkpoint=str(alias), candidate_selected=selection['eligible_for_selection'])
    except Exception as error:
        status('requires inspection; dialogue-retention pipeline stopped', error=str(error))
        raise


if __name__ == '__main__':
    main()
