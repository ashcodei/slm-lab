"""Matched native caption/OCR/object test of the own object-supervised encoder."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from slm_lab.checkpoint_archive import archive_completed
from slm_lab.completed_state_cleanup import retire
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.model import ModelConfig, TinyMultimodalLM
from slm_lab.vision_alignment_experiment import compare_reports
from slm_lab.vision_transfer import language_digest
from train_multimodal_pilot import complete_stage

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    reports = ROOT / 'reports/supervised-vision-alignment-v1'; reports.mkdir(exist_ok=True)
    checkpoints = ROOT / 'checkpoints/supervised-vision-alignment-v1'
    if any(reports.iterdir()) or checkpoints.exists(): raise ValueError('Existing native supervised-vision experiment needs inspection')
    current_sources = capture()
    files = [Path(__file__), *[ROOT / 'src/slm_lab' / (name + '.py') for name in
             ['object_supervision', 'object_supervision_data', 'vision_transfer', 'vision_alignment_experiment',
              'checkpoint_archive', 'completed_state_cleanup', 'object_feature_probe', 'object_qa']]]
    extras = capture(files)

    def status(message, **fields):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
                    'goal_achieved': False, 'production_promoted': False, **fields})

    def guard():
        if capture() != current_sources or capture(files) != extras: raise ValueError('Native alignment implementation/runtime changed')

    def run(command):
        guard()
        subprocess.run(command, cwd=ROOT, check=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '2',
                       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'})
        guard()

    try:
        classification = ROOT / 'reports/object-supervision-v1'
        if json.loads((classification / 'status.json').read_text())['status'] != 'paired supervised-object classification completed; native multimodal capability unverified':
            raise ValueError('Object-supervision experiment not complete')
        result = json.loads((classification / 'comparison.json').read_text())
        if result['paired_parent_bootstrap_95_interval'][0] <= 0:
            raise ValueError('No positive matched classification interval; inspect before native transfer experiment')
        prior = ROOT / 'reports/paired-vision-alignment-v1'; prior_plan = json.loads((prior / 'plan.json').read_text())
        initial = Path(prior_plan['preparation']['branches']['original']['checkpoint'])
        control = ROOT / 'checkpoints/paired-vision-alignment-v1/original/caption/model.pt'
        donor_root = ROOT / 'checkpoints/object-supervision-v1'
        donor = donor_root / 'encoder/model.pt'; frozen = donor_root / 'frozen/model.pt'
        if file_hash(initial) != prior_plan['preparation']['branches']['original']['checkpoint_sha256']:
            raise ValueError('Original matched foundation weights changed')
        base = torch.load(initial, map_location='cpu', weights_only=True, mmap=True)
        learned = torch.load(donor, map_location='cpu', weights_only=True, mmap=True)
        unchanged = torch.load(frozen, map_location='cpu', weights_only=True, mmap=True)
        if learned['steps'] != learned['planned_steps'] or learned['branch'] != 'encoder' or unchanged['branch'] != 'frozen':
            raise ValueError('Incomplete or wrong classification donor')
        if learned['config'] != base['config'] or unchanged['config'] != base['config']:
            raise ValueError('Donor architecture differs')
        expected = {name.removeprefix('vision.'): value for name, value in base['model'].items() if name.startswith('vision.')}
        if (set(learned['vision']) != set(expected) or set(unchanged['vision']) != set(expected)
                or any(not torch.equal(value, unchanged['vision'][name]) for name, value in expected.items())):
            raise ValueError('Classification started from a different encoder than the paired native control')
        if any(learned['vision'][name].shape != value.shape or learned['vision'][name].dtype != value.dtype
               or not torch.isfinite(learned['vision'][name]).all() for name, value in expected.items()):
            raise ValueError('Donor vision tensors invalid')
        if not torch.equal(learned['vision']['project.weight'], expected['project.weight']):
            raise ValueError('Unused language projector was changed during classification')
        with torch.device('meta'): parameters = TinyMultimodalLM(ModelConfig(**base['config'])).parameter_counts()['total']
        if parameters >= 1_000_000_000: raise ValueError('Native experiment exceeds user parameter budget')
        expected_language = language_digest(base['model'])
        checkpoints.mkdir(); root = checkpoints / 'initial'; root.mkdir()
        candidate_initial = root / 'model.pt'
        payload = {key: value for key, value in base.items() if key not in {'optimizer', 'reference_model', 'torch_rng_state', 'cuda_rng_state'}}
        payload['model'] = {**base['model'], **{'vision.' + name: value for name, value in learned['vision'].items()}}
        payload['vision_source'] = 'own original-human-box supervised encoder; classification head not transferred'
        torch.save(payload, candidate_initial); shutil.copy2(initial.parent / 'tokenizer.json', root / 'tokenizer.json')
        def verify(checkpoint):
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
            if language_digest(saved['model']) != expected_language: raise ValueError('Vision alignment changed language/critic tensors')
        verify(candidate_initial)
        receipt = checkpoints / 'preparation.json'
        atomic_json(receipt, {'branches': {'candidate': {'checkpoint': str(candidate_initial), 'checkpoint_sha256': file_hash(candidate_initial)}},
                    'language_state_sha256': expected_language, 'source_original_checkpoint_sha256': file_hash(initial),
                    'object_supervised_donor_sha256': file_hash(donor), 'classification_head_transferred': False,
                    'parameters': parameters, 'goal_achieved': False})
        del base, learned, unchanged, payload, expected
        data = ROOT / 'data/paired-vision-alignment-v1'
        if file_hash(data / 'train.jsonl') != prior_plan['data_sha256']: raise ValueError('Exact matched captions changed')
        for family in ['vision', 'ocr', 'objects']:
            source = prior / ('original-after-' + family + '.json')
            report = json.loads(source.read_text())
            if report['checkpoint_sha256'] != file_hash(control) or report['evaluation_provenance'] != current_sources:
                raise ValueError('Original native control report differs from exact current protocol')
            shutil.copy2(source, reports / ('control-' + family + '.json'))
        plan = {'source_foundation_updates': 2000, 'initial_preparation_sha256': file_hash(receipt),
                'classification_comparison_sha256': file_hash(classification / 'comparison.json'),
                'control_checkpoint_sha256': file_hash(control), 'caption_data_sha256': file_hash(data / 'train.jsonl'),
                'stages': prior_plan['stages'], 'training_images': prior_plan['training_images'],
                'seed': 42, 'parameters': parameters, 'language_state_sha256': expected_language,
                'evaluator_provenance': current_sources, 'extra_sources_runtime': extras,
                'classification_head_transferred': False, 'goal_achieved': False, 'production_promoted': False,
                'limitations': ['Same own partially trained language snapshot, captions/budgets and reused native development cohorts as original control.',
                                'Auxiliary crop classification improvement cannot establish native image answers or high-end capability.',
                                'No automatic production promotion; initial/final research weights are losslessly archived after use.']}
        atomic_json(reports / 'plan.json', plan)
        current = candidate_initial
        for stage in plan['stages']:
            directory = checkpoints / stage['name']
            status('training matched native alignment for supervised encoder', stage=stage['name'], steps=stage['steps'])
            run([sys.executable, '-u', '-m', 'slm_lab.training', stage['stage'], '--checkpoint', str(current),
                 '--tokenizer', str(current.parent / 'tokenizer.json'), '--data', str(data / 'train.jsonl'), '--output', str(directory),
                 '--trainable', 'vision', '--steps', str(stage['steps']), '--batch-size', str(stage['batch_size']),
                 '--gradient-accumulation', str(stage['accumulation']), '--learning-rate', str(stage['learning_rate']),
                 '--image-size', str(stage['image_size']), '--seed', '42', '--device', 'cuda', '--threads', '1',
                 '--gradient-checkpointing', '--checkpoint-every', '200', '--keep-checkpoints', '1', '--log-every', '10'])
            specification = {'stage': stage['stage'], 'steps': stage['steps'], 'trainable': 'vision',
                             'image_size': stage['image_size'], 'data_sha256': plan['caption_data_sha256']}
            if not complete_stage(directory, specification): raise ValueError('Matched native alignment stage incomplete')
            current = directory / 'model.pt'; verify(current)
            for snapshot in directory.glob('step-*.pt'): snapshot.unlink()
            retire(directory, reports / (stage['name'] + '-completed-state-cleanup.json'))
            if stage['name'] == 'contrastive':
                archive_completed(candidate_initial, control, candidate_initial.with_name('model.pt.archive'), True, receipt)
        for family, module, held, extra in [
            ('vision', 'slm_lab.vision_evaluation', data / 'valid.jsonl', ['--image-size', '448']),
            ('ocr', 'slm_lab.ocr_evaluation', ROOT / 'data/alignment-curriculum-v1/images.valid.jsonl', []),
            ('objects', 'slm_lab.object_evaluation', ROOT / 'data/object-qa-v1/valid.jsonl', [])]:
            status('measuring native supervised-encoder responses', family=family)
            run([sys.executable, '-u', '-m', module, '--checkpoint', str(current), '--data', str(held),
                 '--device', 'cuda', '--output', str(reports / ('candidate-' + family + '.json')), *extra])
        comparison = {family: compare_reports(json.loads((reports / ('control-' + family + '.json')).read_text()),
                         json.loads((reports / ('candidate-' + family + '.json')).read_text()), family)
                      for family in ['vision', 'ocr', 'objects']}
        atomic_json(reports / 'comparison.json', {'native': comparison, 'exact_language_and_critic_preserved': True,
                    'classification_head_transferred': False, 'goal_achieved': False, 'production_promoted': False,
                    'limitations': plan['limitations']})
        for name in ['contrastive', 'caption']:
            path = checkpoints / name / 'model.pt'
            archive_completed(path, control, path.with_name('model.pt.archive'), True)
        status('matched native supervised-vision experiment completed; broader capability unverified')
    except Exception as error:
        status('requires inspection; supervised-vision native experiment stopped', error=str(error)); raise


if __name__ == '__main__': main()
