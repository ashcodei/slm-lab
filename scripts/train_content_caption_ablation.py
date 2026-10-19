"""Own caption-pooling ablation on the exact earlier frozen-language/data budget."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from slm_lab.completed_state_cleanup import retire
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.vision_alignment_experiment import compare_reports
from slm_lab.vision_transfer import language_digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    reports = ROOT / 'reports/content-caption-ablation-v1'; reports.mkdir(exist_ok=True)
    checkpoints = ROOT / 'checkpoints/content-caption-ablation-v1'
    if any(reports.iterdir()) or (checkpoints.exists() and any(checkpoints.iterdir())):
        raise ValueError('Existing ablation requires inspection; no automatic restart')
    def status(message, **fields):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
                                             'goal_achieved': False, 'production_promoted': False, **fields})
    try:
        previous = ROOT / 'reports/paired-vision-alignment-v1'
        if json.loads((previous / 'status.json').read_text())['status'] != 'paired caption alignment and native visual checks completed; broader capability unverified':
            raise ValueError('Original matched vision experiment has not completed')
        original_plan = json.loads((previous / 'plan.json').read_text())
        initial = Path(original_plan['preparation']['branches']['original']['checkpoint'])
        original = ROOT / 'checkpoints/paired-vision-alignment-v1/original/caption/model.pt'
        if file_hash(initial) != original_plan['preparation']['branches']['original']['checkpoint_sha256']:
            raise ValueError('Frozen own initial checkpoint changed')
        data = ROOT / 'data/paired-vision-alignment-v1'
        if file_hash(data / 'train.jsonl') != original_plan['data_sha256']:
            raise ValueError('Original matched training captions changed')
        expected_language = original_plan['preparation']['branches']['original']['language_state_sha256']
        shared = capture()
        extra_names = ['scripts/train_content_caption_ablation.py', 'src/slm_lab/content_caption_training.py',
                       'src/slm_lab/content_caption_evaluation.py', 'src/slm_lab/vision_alignment_experiment.py',
                       'src/slm_lab/vision_transfer.py', 'src/slm_lab/completed_state_cleanup.py', 'src/slm_lab/objectives.py']
        extras = {name: file_hash(ROOT / name) for name in extra_names}
        plan = {'original_experiment_plan_sha256': file_hash(previous / 'plan.json'),
                'initial_checkpoint_sha256': file_hash(initial), 'control_checkpoint_sha256': file_hash(original),
                'language_state_sha256': expected_language, 'training_data_sha256': file_hash(data / 'train.jsonl'),
                'validation_data_sha256': file_hash(data / 'valid.jsonl'), 'training_images': original_plan['training_images'],
                'stages': original_plan['stages'], 'seed': 42, 'physical_gpu': 2, 'trainable': 'vision',
                'caption_feature_pooling': 'content mean excluding BOS/EOS/padding',
                'shared_source_runtime': shared, 'extra_implementations': extras,
                'goal_achieved': False, 'production_promoted': False,
                'scope': 'Change only frozen-caption pooling in contrastive training; same own initial weights/data/budgets.',
                'limitations': ['Own partially trained language snapshot and reused development image tests.',
                               'A better content-feature retrieval score alone cannot prove improved native image use.',
                               'No automatic production promotion or high-end capability claim.']}
        atomic_json(reports / 'plan.json', plan)
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '2', 'OMP_NUM_THREADS': '1',
                       'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}

        def run(command):
            if capture() != shared or any(file_hash(ROOT / name) != digest for name, digest in extras.items()):
                raise ValueError('Ablation implementation/runtime changed')
            subprocess.run(command, cwd=ROOT, env=environment, check=True)

        def verify_language(checkpoint):
            state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
            if language_digest(state['model']) != expected_language:
                raise ValueError('Vision ablation changed exact language/critic tensors')

        verify_language(initial); verify_language(original)
        # Existing native control reports are reusable only with identical source,
        # tokenizer/precision/checkpoint; the new content readout is measured fresh.
        for family in ['vision', 'ocr', 'objects']:
            source = previous / ('original-after-' + family + '.json')
            assessed = json.loads(source.read_text())
            if assessed['checkpoint_sha256'] != plan['control_checkpoint_sha256'] or assessed['evaluation_provenance'] != shared:
                raise ValueError('Stored native control report differs from this protocol/model')
            shutil.copy2(source, reports / ('control-' + family + '.json'))
        status('measuring matched control content-feature retrieval')
        run([sys.executable, '-u', '-m', 'slm_lab.content_caption_evaluation', '--checkpoint', str(original),
             '--data', str(data / 'valid.jsonl'), '--output', str(reports / 'control-content.json'), '--device', 'cuda'])
        current = initial
        for stage in plan['stages']:
            if shutil.disk_usage(ROOT).free < 25 * 1024**3:
                raise ValueError('Insufficient space alongside active primary training')
            directory = checkpoints / stage['name']
            status('training content-caption ablation', stage=stage['name'], steps=stage['steps'])
            if stage['stage'] == 'contrastive':
                command = [sys.executable, '-u', '-m', 'slm_lab.content_caption_training']
            else:
                command = [sys.executable, '-u', '-m', 'slm_lab.training', 'sft', '--trainable', 'vision',
                           '--gradient-accumulation', str(stage['accumulation']), '--gradient-checkpointing',
                           '--threads', '1', '--keep-checkpoints', '1', '--log-every', '10']
            command += ['--checkpoint', str(current), '--tokenizer', str(current.parent / 'tokenizer.json'),
                        '--data', str(data / 'train.jsonl'), '--output', str(directory), '--device', 'cuda',
                        '--steps', str(stage['steps']), '--batch-size', str(stage['batch_size']),
                        '--learning-rate', str(stage['learning_rate']), '--image-size', str(stage['image_size']),
                        '--seed', '42', '--checkpoint-every', '200']
            run(command)
            report = json.loads((directory / 'training.json').read_text())
            if (report['stage'] != stage['stage'] or report['steps'] != stage['steps'] or report['trainable'] != 'vision'
                    or report['data_sha256'] != plan['training_data_sha256'] or report['image_size'] != stage['image_size']
                    or report['truncated_sequences']):
                raise ValueError('Ablation stage completion differs from declared paired budget')
            if stage['stage'] == 'contrastive':
                target = json.loads((directory / 'caption-target.json').read_text())
                if target['implementation_sha256'] != extras['src/slm_lab/content_caption_training.py'] or target['shared_trainer_modified']:
                    raise ValueError('Content target was not the declared isolated implementation')
            current = directory / 'model.pt'; verify_language(current)
            for snapshot in directory.glob('step-*.pt'): snapshot.unlink()
            retire(directory, reports / (stage['name'] + '-completed-state-cleanup.json'))
        status('measuring candidate content-feature retrieval')
        run([sys.executable, '-u', '-m', 'slm_lab.content_caption_evaluation', '--checkpoint', str(current),
             '--data', str(data / 'valid.jsonl'), '--output', str(reports / 'candidate-content.json'), '--device', 'cuda'])
        for family, module, held_out, extra in [
            ('vision', 'slm_lab.vision_evaluation', data / 'valid.jsonl', ['--image-size', '448']),
            ('ocr', 'slm_lab.ocr_evaluation', ROOT / 'data/alignment-curriculum-v1/images.valid.jsonl', ['--image-size', '448']),
            ('objects', 'slm_lab.object_evaluation', ROOT / 'data/object-qa-v1/valid.jsonl', []),
        ]:
            status('measuring native content-caption ablation', family=family)
            run([sys.executable, '-u', '-m', module, '--checkpoint', str(current), '--data', str(held_out),
                 '--output', str(reports / ('candidate-' + family + '.json')), '--device', 'cuda', *extra])
        before = json.loads((reports / 'control-content.json').read_text()); after = json.loads((reports / 'candidate-content.json').read_text())
        for key in ['data_sha256', 'evaluation_provenance', 'image_size', 'caption_pooling', 'tokenizer_sha256', 'inference_precision', 'cases']:
            if before[key] != after[key]: raise ValueError('Content retrieval comparison differs: ' + key)
        native = {}
        for family in ['vision', 'ocr', 'objects']:
            first = json.loads((reports / ('control-' + family + '.json')).read_text())
            second = json.loads((reports / ('candidate-' + family + '.json')).read_text())
            if second['checkpoint_sha256'] != file_hash(current): raise ValueError('Candidate report/model identity differs')
            native[family] = compare_reports(first, second, family)
        atomic_json(reports / 'comparison.json', {'content_retrieval': {'control': before['caption_retrieval'],
                    'candidate': after['caption_retrieval']}, 'native': native, 'exact_language_and_critic_preserved': True,
                    'goal_achieved': False, 'production_promoted': False, 'limitations': plan['limitations']})
        status('content-caption ablation and native checks completed; broader capability unverified')
    except Exception as error:
        status('content-caption ablation stopped; requires inspection', error=str(error))
        raise


if __name__ == '__main__':
    main()
