"""Own dense-description format learning, faster frozen-language vision learning, then QA."""
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time

import torch
from slm_lab.checkpoint_archive import archive_completed
from slm_lab.clevr_evaluation import compare_reports
from slm_lab.completed_state_cleanup import retire
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.masked_visual_pretraining import process_identity
from slm_lab.vision_transfer import language_digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    reports=ROOT/'reports/clevr-dense-learning-v1';checkpoints=ROOT/'checkpoints/clevr-dense-learning-v1'
    if reports.exists() or checkpoints.exists():raise ValueError('Existing dense learning requires inspected resume; preserve prior work')
    preceding=ROOT/'reports/clevr-learning-probe-v1'
    if json.loads((preceding/'status.json').read_text())['status']!='matched original visual reasoning learning probe completed; broader capability unverified':
        raise ValueError('Previous matched study has not completed')
    dense=ROOT/'data/clevr-dense-scenes-v1';manifest=json.loads((dense/'manifest.json').read_text())
    if file_hash(ROOT/'src/slm_lab/clevr_scene_supervision.py')!=manifest['implementation_sha256']:
        raise ValueError('Original dense-target preparation implementation changed')
    for name,digest in manifest['files_sha256'].items():
        if file_hash(dense/name)!=digest:raise ValueError('Original dense scene descriptions changed')
    base=ROOT/'checkpoints/clevr-foundation-baseline-v1/model.pt'
    baseline_path=ROOT/'reports/clevr-foundation-baseline-v1/visual-reasoning.json';baseline=json.loads(baseline_path.read_text())
    vanilla_path=preceding/'joint-visual-reasoning.json';vanilla=json.loads(vanilla_path.read_text())
    if file_hash(base)!=baseline['checkpoint_sha256']:raise ValueError('Own retained foundation base changed')
    eval_files=[ROOT/name for name in baseline['evaluation_provenance']['files_sha256']]
    if capture(eval_files)!=baseline['evaluation_provenance'] or vanilla['evaluation_provenance']!=baseline['evaluation_provenance']:
        raise ValueError('Native visual baseline evaluators differ')
    files=[*eval_files,Path(__file__),ROOT/'src/slm_lab/clevr_scene_supervision.py',ROOT/'src/slm_lab/vision_transfer.py',
           ROOT/'src/slm_lab/completed_state_cleanup.py',ROOT/'src/slm_lab/checkpoint_archive.py']
    loaded=capture(files);reports.mkdir();checkpoints.mkdir()
    stages=[('format','language',500,2e-5,dense/'train.jsonl'),('vision','vision',1000,2e-4,dense/'train.jsonl'),
            ('questions','all',1250,2e-5,ROOT/'data/clevr-reasoning-v1/train.jsonl')]
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'base_checkpoint_sha256':baseline['checkpoint_sha256'],
          'dense_manifest_sha256':file_hash(dense/'manifest.json'),'baseline_report_sha256':file_hash(baseline_path),
          'vanilla_joint_report_sha256':file_hash(vanilla_path),'seed':42,'batch_size':2,'gradient_accumulation':4,
          'world_size':1,'image_size':448,'device':'cuda:2','expected_dense_format_targets':311386,'expected_vision_targets':622772,
          'stages':[{'name':name,'trainable':trainable,'steps':steps,'learning_rate':lr,'data_sha256':file_hash(data),
                     'record_exposures':steps*8} for name,trainable,steps,lr,data in stages],
          'source_provenance':loaded,'goal_achieved':False,'policy_promoted':False,
          'scope':'Additional dense labels and38.8x approximate supervised-target budget versus prior short-QA study; exploratory conditioning hypothesis, not matched-budget causal superiority.'}
    atomic_json(reports/'plan.json',plan)

    def status(message,**values):
        atomic_json(reports/'status.json',{'status':message,'updated_at':datetime.now(timezone.utc).isoformat(),'goal_achieved':False,**values})
    def run(arguments):
        if capture(files)!=loaded:raise ValueError('Loaded dense-learning implementation changed')
        subprocess.run(arguments,check=True,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'})

    try:
        receipts={};current=base
        for index,(name,trainable,steps,lr,data) in enumerate(stages):
            while shutil.disk_usage(ROOT).free < 30*1024**3:
                if capture(files)!=loaded or process_identity(71312) is None:raise ValueError('Disk wait requires unchanged sources and verified existing main trainer')
                status('waiting for free disk before starting another dense-learning stage',stage=name);time.sleep(15)
            directory=checkpoints/name
            status('learning native dense scene descriptions' if name!='questions' else 'learning original visual questions after dense grounding',
                   stage=name,trainable=trainable,steps=steps,learning_rate=lr)
            previous_sha=file_hash(current)
            run([sys.executable,'-u','-m','slm_lab.training','sft','--checkpoint',str(current),'--tokenizer',str(current.parent/'tokenizer.json'),
                 '--data',str(data),'--output',str(directory),'--trainable',trainable,'--steps',str(steps),'--batch-size','2',
                 '--gradient-accumulation','4','--learning-rate',str(lr),'--seed','42','--image-size','448','--device','cuda:2',
                 '--threads','1','--gradient-checkpointing','--checkpoint-every','0','--log-every','25'])
            training=json.loads((directory/'training.json').read_text())
            actual=sum(item['global_target_tokens'] for item in training['metrics'])
            if (training['stage']!='sft' or training['steps']!=steps or training['planned_steps']!=steps or training['trainable']!=trainable
                    or training['world_size']!=1 or training['truncated_sequences']!=0 or training['data_sha256']!=plan['stages'][index]['data_sha256']
                    or [item['step'] for item in training['metrics']]!=list(range(1,steps+1))):
                raise ValueError('Declared dense learning stage/budget did not complete')
            if name in {'format','vision'} and actual!=plan['expected_dense_format_targets' if name=='format' else 'expected_vision_targets']:
                raise ValueError('Dense source targets were repeated/lost beyond the declared whole passes')
            original=torch.load(current,map_location='cpu',weights_only=True,mmap=True)
            candidate=torch.load(directory/'model.pt',map_location='cpu',weights_only=True,mmap=True)
            preservation={}
            if name=='format':
                keys=[key for key in original['model'] if key.startswith('vision.')]
                if any(not torch.equal(original['model'][key],candidate['model'][key]) for key in keys):raise ValueError('Format warmup changed frozen vision')
                preservation['exact_frozen_vision_tensors']=len(keys)
            if name=='vision':
                if language_digest(original['model'])!=language_digest(candidate['model']):raise ValueError('Faster image-only learning changed language/critic')
                preservation['exact_language_and_critic_sha256']=language_digest(candidate['model'])
            del original,candidate
            receipts[name]={'source_checkpoint_sha256':previous_sha,'model_sha256':file_hash(directory/'model.pt'),
                            'actual_target_exposures':actual,'record_exposures':steps*8,'preservation':preservation}
            atomic_json(reports/(name+'-stage-receipt.json'),receipts[name])
            retire(directory,reports/(name+'-completed-state-cleanup.json'))
            if name=='vision':
                archive=archive_completed(checkpoints/'format/model.pt',directory/'model.pt',checkpoints/'format/model.pt.archive',remove_original=True)
                atomic_json(reports/'format-exact-archive.json',{key:value for key,value in archive.items() if key!='segments'})
            current=directory/'model.pt'
        status('evaluating native questions after richer scene supervision')
        output=reports/'visual-reasoning.json'
        run([sys.executable,'-u','-m','slm_lab.clevr_evaluation','--checkpoint',str(current),'--data',str(ROOT/'data/clevr-reasoning-v1/valid.jsonl'),
             '--output',str(output),'--device','cuda:2'])
        candidate=json.loads(output.read_text());guard=compare_reports(vanilla,candidate)
        report={'base':baseline['overall'],'vanilla_joint':vanilla['overall'],'dense_staged_candidate':candidate['overall'],
                'vanilla_retention_guard':guard,'stages':receipts,'policy_promoted':False,'goal_achieved':False,
                'scope':'Extra supervision/budget exploratory probe; no real-photo, code/knowledge retention, causal method superiority or high-end capability claim.'}
        atomic_json(reports/'comparison.json',report)
        status('dense native scene learning/evaluation completed; broader capability unverified')
    except Exception as error:
        status('dense scene learning failed; inspect without restarting live work',error=str(error));raise


if __name__=='__main__':main()
