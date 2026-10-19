"""Continue the from-scratch model on diverse images after text training/evaluation."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json,jsonl

ROOT=Path(__file__).resolve().parents[1]
STAGES=[
    {'name':'instruction','stage':'sft','file':'instruction.train.jsonl','passes':1,
     'batch':2,'accumulation':4,'trainable':'language','learning_rate':2e-5},
    {'name':'contrastive','stage':'contrastive','file':'contrastive.train.jsonl','passes':3,
     'batch':8,'accumulation':1,'trainable':'vision','learning_rate':1e-4},
    {'name':'image-reading','stage':'sft','file':'images.train.jsonl','passes':2,
     'batch':2,'accumulation':2,'trainable':'vision','learning_rate':1e-4},
    {'name':'joint-instruction-images','stage':'sft','file':'joint.train.jsonl','passes':1,
     'batch':2,'accumulation':2,'trainable':'all','learning_rate':1e-5,'image_size':448},
]


def complete_stage(directory,stage):
    path=directory/'training.json'
    if not path.exists():return False
    result=json.loads(path.read_text())
    if (result['steps']!=stage['steps'] or result['data_sha256']!=stage['data_sha256']
        or result['stage']!=stage['stage'] or result['trainable']!=stage['trainable']):
        raise ValueError('Completed stage differs from its recorded plan: '+str(directory))
    if result.get('image_size',224)!=stage.get('image_size',224):raise ValueError('Completed stage image resolution differs')
    if not (directory/'model.pt').exists():raise ValueError('Completed stage checkpoint missing')
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wait',action='store_true')
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    os.chdir(ROOT);reports=ROOT/'reports/multimodal-pilot-v1';reports.mkdir(exist_ok=True)
    status_path=reports/'status.json'
    def status(state,**fields):
        atomic_json(status_path,{'status':state,'updated_at':datetime.now(timezone.utc).isoformat(),**fields})
    curriculum=ROOT/'data/alignment-curriculum-v1';prerequisites={
        'text training':ROOT/'checkpoints/broad-public-pilot-v1/training.json',
        'post-text evaluation':ROOT/'reports/posttrain-evaluation-v1/status.json',
        'diverse image curriculum':curriculum/'manifest.json'}
    while True:
        missing=[]
        for label,path in prerequisites.items():
            if not path.exists():missing.append(label);continue
            value=json.loads(path.read_text())
            if label=='post-text evaluation' and value.get('status')!='evaluation completed':missing.append(label)
            if label=='diverse image curriculum' and value.get('status')!='prepared; not yet trained':missing.append(label)
        if not missing:break
        status('waiting for completed text training/evaluation and diverse image data',pending=missing)
        if not args.wait:raise ValueError('Prerequisites incomplete: '+', '.join(missing))
        time.sleep(30)
    manifest=json.loads((curriculum/'manifest.json').read_text())
    # Verify every prepared output against the audit manifest before consuming it.
    for name,digest in manifest['files_sha256'].items():
        if file_hash(curriculum/name)!=digest:raise ValueError('Curriculum file changed: '+name)
    questions=ROOT/'data/knowledge-eval-v1/questions.jsonl'
    if file_hash(questions)!=manifest['benchmark_questions_sha256']:raise ValueError('Reserved knowledge questions changed')
    with (curriculum/'joint.train.jsonl').open('w') as output:
        for name in ['instruction.train.jsonl','images.train.jsonl']:
            with (curriculum/name).open() as source:
                for line in source:output.write(line)
    stages=[]
    for template in STAGES:
        source=curriculum/template['file'];records=sum(1 for _ in jsonl(source))
        batch=4*template['batch']*template['accumulation']
        if records<batch:raise ValueError('Too few records for global batch: '+template['name'])
        stages.append({**template,'records':records,'global_batch':batch,
                       'steps':math.ceil(records*template['passes']/batch),'data_sha256':file_hash(source)})
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'stages':stages,
          'curriculum_manifest_sha256':file_hash(curriculum/'manifest.json'),
          'starting_checkpoint':str(ROOT/'checkpoints/broad-public-pilot-v1/model.pt'),
          'devices':'4 local L40 GPUs','external_pretrained_weights':False,
          'evaluation_order':'After all listed training stages; text-pilot evaluation completes before these stages begin.',
          'goal_achieved':False,'limitations':['Bounded curriculum and repeated passes; broad image/OCR competence is not established.',
                                            'Human-feedback policy training is a subsequent experiment, not included in this visual pilot.']}
    atomic_json(reports/'plan.json',plan)
    if args.prepare_only:status('prepared; training not launched',plan=str(reports/'plan.json'));return
    checkpoint=Path(plan['starting_checkpoint']);previous=None
    environment={**os.environ,'OMP_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
    try:
        for stage in stages:
            directory=ROOT/'checkpoints/multimodal-pilot-v1'/stage['name']
            if not complete_stage(directory,stage):
                if directory.exists() and any(directory.iterdir()):
                    raise ValueError('Incomplete stage needs checkpoint inspection/resume before retry: '+str(directory))
                if shutil.disk_usage(ROOT).free<22*1024**3:raise ValueError('Insufficient free disk for the next stage and atomic checkpoints')
                command=[str(ROOT/'.venv/bin/torchrun'),'--standalone','--nproc_per_node=4','-m','slm_lab.training',stage['stage'],
                    '--checkpoint',str(checkpoint),'--tokenizer',str(checkpoint.parent/'tokenizer.json'),
                    '--data',str(curriculum/stage['file']),'--output',str(directory),
                    '--steps',str(stage['steps']),'--batch-size',str(stage['batch']),
                    '--gradient-accumulation',str(stage['accumulation']),'--trainable',stage['trainable'],
                    '--learning-rate',str(stage['learning_rate']),'--device','cuda','--threads','1',
                    '--gradient-checkpointing','--log-every','10','--checkpoint-every','100','--keep-checkpoints','1']
                if stage.get('image_size'):command.extend(['--image-size',str(stage['image_size'])])
                status('training',stage=stage['name'],planned_steps=stage['steps'],checkpoint=str(checkpoint))
                subprocess.run(command,check=True,env=environment,cwd=ROOT)
                if not complete_stage(directory,stage):raise ValueError('Stage did not complete its recorded plan')
            checkpoint=directory/'model.pt'
            # Keep each stage's inference checkpoint and the newest optimizer state.
            # Only remove redundant optimizer snapshots after its successor succeeds.
            if previous is not None:
                for path in [previous/'training-state.pt',*previous.glob('step-*.pt')]:path.unlink(missing_ok=True)
            previous=directory
        for path in previous.glob('step-*.pt'):path.unlink(missing_ok=True)
        evaluations=[
            ('visual grounding at 224px','slm_lab.vision_evaluation',[
                '--data',str(curriculum/'contrastive.valid.jsonl'),'--image-size','224','--output',str(reports/'vision-224.json')]),
            ('visual grounding','slm_lab.vision_evaluation',[
                '--data',str(curriculum/'contrastive.valid.jsonl'),'--output',str(reports/'vision.json')]),
            ('scene and document OCR at 224px','slm_lab.ocr_evaluation',[
                '--data',str(curriculum/'images.valid.jsonl'),'--image-size','224','--output',str(reports/'ocr-224.json')]),
            ('scene and document OCR','slm_lab.ocr_evaluation',[
                '--data',str(curriculum/'images.valid.jsonl'),'--output',str(reports/'ocr.json')]),
            ('code syntax, correctness and efficiency','slm_lab.code_quality',[
                '--output',str(reports/'code-quality.json')]),
            ('domain knowledge','slm_lab.knowledge_evaluation',[
                '--data',str(questions),'--output',str(reports/'knowledge.json')])]
        for label,module,options in evaluations:
            status('evaluating after multimodal training',task=label,checkpoint=str(checkpoint))
            subprocess.run([sys.executable,'-u','-m',module,'--checkpoint',str(checkpoint),'--device','cuda',*options],
                           check=True,cwd=ROOT,env=environment)
        atomic_json(reports/'summary.json',{'checkpoint':str(checkpoint),'checkpoint_sha256':file_hash(checkpoint),
            'curriculum_manifest_sha256':plan['curriculum_manifest_sha256'],
            'reserved_knowledge_questions_sha256':manifest['benchmark_questions_sha256'],
            'alignment_contamination_audit':str(curriculum/'manifest.json'),
            'evaluations':{name:str(reports/name) for name in ['vision.json','vision-224.json','ocr.json','ocr-224.json','code-quality.json','knowledge.json']},
            'goal_achieved':False,'high_end_llm_parity':'Not established',
            'pending':['Larger and more representative training mixtures','Human-feedback policy training',
                       'High-resolution visual understanding','Broader independent capability and comparison tests']})
        status('visual pilot training and evaluation completed; broad capability unverified',checkpoint=str(checkpoint))
    except Exception as error:
        status('requires inspection; pipeline stopped',error=str(error),checkpoint=str(checkpoint));raise

if __name__=='__main__':main()
