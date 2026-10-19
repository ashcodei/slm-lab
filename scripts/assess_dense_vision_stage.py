"""Assess the frozen-language vision stage once its actual training receipt exists."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.masked_visual_pretraining import process_identity

ROOT=Path(__file__).resolve().parents[1]


def main():
    reports=ROOT/'reports/clevr-dense-learning-v1';output=reports/'vision-scene-generation.json'
    if output.exists():raise ValueError('Preserve prior assessment; do not rerun silently')
    pid=288842;identity=process_identity(pid)
    if identity is None or b'scripts/train_dense_scene_probe.py' not in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0'):
        raise ValueError('Dense vision controller is not verified live')
    files=[ROOT/name for name in capture()['files_sha256']]+[ROOT/'src/slm_lab/clevr_scene_evaluation.py',
           ROOT/'src/slm_lab/clevr_scene_supervision.py',Path(__file__)]
    loaded=capture(files);receipt_path=reports/'vision-stage-receipt.json'
    atomic_json(reports/'vision-description-assessment.status.json',{'status':'waiting for verified live dense vision stage',
                'controller_pid':pid,'controller_start_ticks':identity,'goal_achieved':False})
    while not receipt_path.exists():
        if process_identity(pid)!=identity:raise ValueError('Dense controller ended without completed vision receipt')
        if capture(files)!=loaded:raise ValueError('Scene assessment implementation changed while waiting')
        time.sleep(15)
    receipt=json.loads(receipt_path.read_text());checkpoint=ROOT/'checkpoints/clevr-dense-learning-v1/vision/model.pt'
    if (receipt['model_sha256']!=file_hash(checkpoint) or receipt['actual_target_exposures']!=622772
            or not receipt['preservation'].get('exact_language_and_critic_sha256')):
        raise ValueError('Vision-only stage weights/budget/preservation differ')
    if capture(files)!=loaded:raise ValueError('Scene assessment implementation changed before inference')
    atomic_json(reports/'vision-description-assessment.status.json',{'status':'evaluating original held-out scene descriptions','goal_achieved':False})
    subprocess.run([sys.executable,'-u','-m','slm_lab.clevr_scene_evaluation','--checkpoint',str(checkpoint),
                    '--output',str(output),'--device','cuda:2'],check=True,cwd=ROOT,
                   env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'})
    atomic_json(reports/'vision-description-assessment.status.json',{'status':'vision-only scene descriptions assessed; broader capability unverified','goal_achieved':False})


if __name__=='__main__':main()
