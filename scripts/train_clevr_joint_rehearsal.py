"""Assess simultaneous native descriptions/QA after observed sequential forgetting."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from slm_lab.clevr_evaluation import compare_reports
from slm_lab.clevr_joint_rehearsal import compare_scene_reports
from slm_lab.completed_state_cleanup import retire
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.masked_visual_pretraining import process_identity

ROOT = Path(__file__).resolve().parents[1]
DONE = 'joint native scene/QA rehearsal assessed; broader capability unverified'


def main():
    reports = ROOT / 'reports/clevr-joint-rehearsal-v1'
    checkpoints = ROOT / 'checkpoints/clevr-joint-rehearsal-v1'
    if reports.exists() or checkpoints.exists():
        raise ValueError('Preserve existing rehearsal work; inspect any live process before resuming')
    old = ROOT / 'reports/clevr-dense-learning-v1'
    if json.loads((old / 'status.json').read_text())['status'] != 'dense native scene learning/evaluation completed; broader capability unverified':
        raise ValueError('Prior staged study is not terminal')
    data = ROOT / 'data/clevr-joint-rehearsal-v1'
    manifest = json.loads((data / 'manifest.json').read_text())
    base = ROOT / 'checkpoints/clevr-dense-learning-v1/vision/model.pt'
    receipt = json.loads((old / 'vision-stage-receipt.json').read_text())
    if file_hash(base) != receipt['model_sha256']:
        raise ValueError('Own vision checkpoint changed')
    if file_hash(data / 'train.jsonl') != manifest['files_sha256']['train.jsonl']:
        raise ValueError('Prepared rehearsal records changed')
    if file_hash(ROOT / 'src/slm_lab/clevr_joint_rehearsal.py') != manifest['implementation_sha256']:
        raise ValueError('Rehearsal preparation implementation changed')
    for path, digest in manifest['source_files_sha256'].items():
        if file_hash(path) != digest:
            raise ValueError('Original rehearsal source changed')
    scene_before = json.loads((old / 'vision-scene-generation.json').read_text())
    qa_only = json.loads((old / 'visual-reasoning.json').read_text())
    scene_qa_only = json.loads((old / 'questions-scene-generation.json').read_text())
    if scene_before['checkpoint_sha256'] != receipt['model_sha256']:
        raise ValueError('Scene baseline belongs to a different own checkpoint')
    for report in [scene_before, qa_only, scene_qa_only]:
        files = [ROOT / name for name in report['evaluation_provenance']['files_sha256']]
        if capture(files) != report['evaluation_provenance']:
            raise ValueError('Saved baseline assessment implementation/runtime changed')
    pinned = [ROOT / name for name in capture()['files_sha256']] + [
        Path(__file__), ROOT / 'src/slm_lab/clevr_evaluation.py',
        ROOT / 'src/slm_lab/clevr_scene_evaluation.py', ROOT / 'src/slm_lab/clevr_scene_supervision.py',
        ROOT / 'src/slm_lab/clevr_joint_rehearsal.py', ROOT / 'src/slm_lab/completed_state_cleanup.py']
    loaded = capture(pinned)
    reports.mkdir(); checkpoints.mkdir()
    main_identity = process_identity(71312)
    if main_identity != '3964801':
        raise ValueError('Expected main text/code trainer identity changed')
    steps = manifest['steps_one_pass_batch2_accumulation4']
    assert steps == 1750
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'base_checkpoint_sha256': receipt['model_sha256'],
            'data_manifest_sha256': file_hash(data / 'manifest.json'), 'data_sha256': manifest['files_sha256']['train.jsonl'],
            'family_examples': manifest['family_examples'], 'family_target_tokens': manifest['family_target_tokens'],
            'actual_target_exposures_expected': manifest['target_tokens'], 'steps': steps, 'seed': 42,
            'batch_size': 2, 'gradient_accumulation': 4, 'world_size': 1, 'image_size': 448,
            'learning_rate': 2e-5, 'trainable': 'all', 'device': 'cuda:2',
            'qa_only_comparator_sha256': file_hash(old / 'visual-reasoning.json'),
            'scene_baseline_sha256': file_hash(old / 'vision-scene-generation.json'),
            'source_provenance': loaded, 'minimum_free_disk_GiB_before_training': 32,
            'storage_budget': '12GiB candidate checkpoint/state +12GiB main snapshot reserve +8GiB remaining.',
            'policy_promoted': False, 'goal_achieved': False,
            'limitations': manifest['limitations'],
            'scope': 'Additional-budget native task-retention intervention, with matched before/after image controls; no causal superiority or high-end capability claim.'}
    atomic_json(reports / 'plan.json', plan)

    def status(message, **values):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
                                            'goal_achieved': False, **values})

    def run(arguments):
        if capture(pinned) != loaded:
            raise ValueError('Loaded rehearsal implementation/runtime changed')
        subprocess.run(arguments, check=True, cwd=ROOT,
                       env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                            'TOKENIZERS_PARALLELISM': 'false'})

    def assess_qa(checkpoint, output):
        run([sys.executable, '-u', '-m', 'slm_lab.clevr_evaluation', '--checkpoint', str(checkpoint),
             '--data', str(ROOT / 'data/clevr-reasoning-v1/valid.jsonl'), '--output', str(output), '--device', 'cuda:2'])
        return json.loads(output.read_text())

    try:
        status('assessing native questions before simultaneous rehearsal')
        qa_before = assess_qa(base, reports / 'before-visual-reasoning.json')
        while shutil.disk_usage(ROOT).free < 32 * 1024**3:
            if capture(pinned) != loaded or process_identity(71312) != main_identity:
                raise ValueError('Disk wait requires unchanged sources and verified live main trainer')
            status('waiting for candidate and main snapshot disk reserve')
            time.sleep(15)
        if file_hash(base) != receipt['model_sha256'] or file_hash(data / 'train.jsonl') != plan['data_sha256']:
            raise ValueError('Training base/input changed since admission')
        status('learning interleaved native descriptions and questions', steps=steps)
        directory = checkpoints / 'joint'
        run([sys.executable, '-u', '-m', 'slm_lab.training', 'sft', '--checkpoint', str(base),
             '--tokenizer', str(base.parent / 'tokenizer.json'), '--data', str(data / 'train.jsonl'),
             '--output', str(directory), '--trainable', 'all', '--steps', str(steps), '--batch-size', '2',
             '--gradient-accumulation', '4', '--learning-rate', '2e-5', '--seed', '42', '--image-size', '448',
             '--device', 'cuda:2', '--threads', '1', '--gradient-checkpointing', '--checkpoint-every', '0', '--log-every', '25'])
        training = json.loads((directory / 'training.json').read_text())
        actual = sum(item['global_target_tokens'] for item in training['metrics'])
        if (training['steps'] != steps or training['planned_steps'] != steps or training['stage'] != 'sft'
                or training['trainable'] != 'all' or training['world_size'] != 1
                or training['data_sha256'] != plan['data_sha256'] or training['truncated_sequences'] != 0
                or actual != manifest['target_tokens']
                or [item['step'] for item in training['metrics']] != list(range(1, steps + 1))):
            raise ValueError('Declared whole-pass rehearsal exposure/budget did not complete')
        atomic_json(reports / 'stage-receipt.json', {'model_sha256': file_hash(directory / 'model.pt'),
                    'source_checkpoint_sha256': receipt['model_sha256'], 'record_exposures': steps * 8,
                    'actual_target_exposures': actual, 'goal_achieved': False})
        retire(directory, reports / 'completed-state-cleanup.json')
        status('assessing native QA and free scene descriptions after joint rehearsal')
        qa_after = assess_qa(directory / 'model.pt', reports / 'after-visual-reasoning.json')
        run([sys.executable, '-u', '-m', 'slm_lab.clevr_scene_evaluation', '--checkpoint', str(directory / 'model.pt'),
             '--output', str(reports / 'after-scene-generation.json'), '--device', 'cuda:2'])
        scene_after = json.loads((reports / 'after-scene-generation.json').read_text())
        result = {'native_QA_from_vision_base': compare_reports(qa_before, qa_after),
                  'native_QA_against_QA_only': compare_reports(qa_only, qa_after),
                  'scene_retention_from_vision_base': compare_scene_reports(scene_before, scene_after),
                  'scene_against_QA_only': compare_scene_reports(scene_qa_only, scene_after),
                  'QA_before': qa_before['overall'], 'QA_after': qa_after['overall'],
                  'QA_only': qa_only['overall'], 'scene_after': scene_after['overall'],
                  'policy_promoted': False, 'goal_achieved': False, 'limitations': plan['limitations'], 'scope': plan['scope']}
        atomic_json(reports / 'comparison.json', result)
        status(DONE)
    except Exception as error:
        status('joint scene/QA rehearsal failed; inspect without restarting live work', error=str(error))
        raise


if __name__ == '__main__':
    main()
