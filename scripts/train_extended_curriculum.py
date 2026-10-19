"""Continue from our own policy on larger public language/code data, then realign vision."""
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json,jsonl
from slm_lab.mix_corpora import mix
from slm_lab.snapshot_corpus import freeze as freeze_wikipedia
from train_multimodal_pilot import complete_stage

ROOT=Path(__file__).resolve().parents[1]


def foundation_warmup(base,code,wiki,reports,status):
    """Use a fixed completed-part snapshot while the rest of Wikipedia collects."""
    snapshot=ROOT/'data/wikipedia-foundation-warmup-v1'
    freeze_wikipedia(wiki,snapshot)
    packed=ROOT/'data/expanded-foundation-warmup-v1'
    if not (packed/'corpus.json').exists():
        mixture=mix([('documented-code',code,None),('encyclopedia',snapshot,None)],packed)
    else:
        mixture=json.loads((packed/'corpus.json').read_text())
        for component in mixture['components']:
            if file_hash(Path(component['path'])/'corpus.json')!=component['manifest_sha256']:raise ValueError('Frozen warmup component changed')
        for values in mixture['splits'].values():
            for shard in values['shards']:
                if file_hash(packed/shard['path'])!=shard['sha256']:raise ValueError('Frozen warmup shard changed')
    spec={'stage':'pretrain','trainable':'language','image_size':448,
          'steps':(mixture['splits']['train']['tokens']-1)//262080,'data_sha256':file_hash(packed/'corpus.json')}
    if spec['steps']<1:raise ValueError('Warmup source budget is empty')
    plan={**spec,'base_checkpoint':str(base),'base_checkpoint_sha256':file_hash(base),
          'snapshot_manifest_sha256':file_hash(snapshot/'corpus.json'),
          'train_tokens':mixture['splits']['train']['tokens'],'passes':1,
          'planned_target_tokens':spec['steps']*262080,
          'learning_rate':1e-4,'batch_size':1,'gradient_accumulation':16,
          'limitations':['Warmup tokens recur in the final larger corpus; later training counts as repeated exposure.',
                         'A training stage does not establish useful or high-end capability.']}
    path=reports/'foundation-warmup-plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Foundation warmup differs from its recorded plan')
    atomic_json(path,plan);directory=ROOT/'checkpoints/extended-curriculum-v1/foundation-warmup'
    if not complete_stage(directory,spec):
        if shutil.disk_usage(ROOT).free<25*1024**3:raise ValueError('Insufficient free disk for foundation warmup checkpoints')
        snapshots=sorted(directory.glob('step-*.pt'));resume=bool(snapshots)
        if directory.exists() and any(directory.iterdir()) and not resume:raise ValueError('Incomplete warmup needs inspection; no resumable snapshot')
        command=[str(ROOT/'.venv/bin/torchrun'),'--standalone','--nproc_per_node=4','-m','slm_lab.training','pretrain',
                 '--checkpoint',str(snapshots[-1] if resume else base),'--tokenizer',str(base.parent/'tokenizer.json'),
                 '--data',str(packed),'--output',str(directory),'--trainable','language',
                 '--steps',str(spec['steps']),'--batch-size','1','--gradient-accumulation','16','--learning-rate','1e-4',
                 '--image-size','448','--device','cuda','--threads','1','--gradient-checkpointing',
                 '--checkpoint-every','1000','--keep-checkpoints','1','--log-every','10']
        if resume:command.append('--resume')
        status('training expanded foundation on frozen completed source parts',stage='foundation-warmup',planned_steps=spec['steps'],resume=resume)
        env={**os.environ,'OMP_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
        subprocess.run(command,check=True,cwd=ROOT,env=env)
        if not complete_stage(directory,spec):raise ValueError('Foundation warmup did not complete')
    for old in directory.glob('step-*.pt'):old.unlink(missing_ok=True)
    checkpoint=directory/'model.pt'
    atomic_json(reports/'foundation-warmup-completed.json',{'checkpoint':str(checkpoint),
                'checkpoint_sha256':file_hash(checkpoint),'completed_steps':spec['steps'],
                'plan_sha256':file_hash(path),'goal_achieved':False})
    return checkpoint,plan


def main():
    os.chdir(ROOT);reports=ROOT/'reports/extended-curriculum-v1';reports.mkdir(exist_ok=True)
    def status(state,**fields):
        atomic_json(reports/'status.json',{'status':state,'updated_at':datetime.now(timezone.utc).isoformat(),**fields})
    code=ROOT/'data/code-search-expanded-v1';wiki=ROOT/'data/wikipedia-expanded-v1'
    prior=ROOT/'reports/human-feedback-pilot-v1/status.json'
    lettering=ROOT/'reports/lettering-experiment-v1'
    objects=ROOT/'data/object-qa-v1'
    verified=ROOT/'data/verified-code-curriculum-v1'
    public_code=ROOT/'data/public-code-assessment-v1'
    math_data=ROOT/'data/math-reasoning-v2'
    native_readiness=ROOT/'reports/cpp-reference-readiness.json'
    while True:
        missing=[str(p) for p in [code/'corpus.json',wiki/'collection-config.json'] if not p.exists()]
        if not any((wiki/'parts').glob('*/corpus.json')):missing.append('committed Wikipedia source parts')
        if not prior.exists() or json.loads(prior.read_text()).get('status')!='human-feedback pilot training/evaluation completed; broader goal unverified':missing.append('completed current human-feedback pilot')
        if not (lettering/'status.json').exists() or json.loads((lettering/'status.json').read_text()).get('status')!='visual lettering experiment completed; selection recorded':missing.append('completed lettering experiment and checkpoint selection')
        if not (objects/'manifest.json').exists():missing.append('completed object/count/spatial question curriculum')
        if not (verified/'manifest.json').exists():missing.append('completed published-example-verified code curriculum')
        if not (public_code/'manifest.json').exists():missing.append('prepared broader public-code assessment')
        if not (math_data/'manifest.json').exists():missing.append('prepared original-human math reasoning curriculum')
        if not native_readiness.exists():missing.append('validated native C++ reference assessment')
        if not missing:break
        status('waiting for expanded source corpora and completed current pilots',pending=missing);time.sleep(30)
    selection=json.loads((lettering/'selection.json').read_text())
    base=Path(selection['selected_checkpoint']).resolve()
    expected=ROOT/('checkpoints/lettering-experiment-v1/joint-reading/model.pt' if selection['promote_candidate'] else 'checkpoints/human-feedback-pilot-v1/dpo/model.pt')
    if base!=expected.resolve() or file_hash(base)!=selection['selected_checkpoint_sha256']:raise ValueError('Selected own-policy checkpoint changed')
    object_manifest=json.loads((objects/'manifest.json').read_text())
    for name in ['train.jsonl','valid.jsonl']:
        if file_hash(objects/name)!=object_manifest['files_sha256'][name]:raise ValueError('Object question records changed')
    verified_manifest=json.loads((verified/'manifest.json').read_text())
    if file_hash(verified/'train.jsonl')!=verified_manifest['train_sha256']:raise ValueError('Verified code instruction records changed')
    public_code_manifest=json.loads((public_code/'manifest.json').read_text())
    if file_hash(public_code/'questions.jsonl')!=public_code_manifest['questions_sha256']:raise ValueError('Public code assessment changed')
    cpp_ready=json.loads(native_readiness.read_text())
    if cpp_ready['tasks']!=12 or any(not row['result']['functional_passed'] or not row['result']['efficiency_completed'] for row in cpp_ready['references']):raise ValueError('Native C++ references are not validated')
    math_manifest=json.loads((math_data/'manifest.json').read_text())
    for name,digest in math_manifest['files_sha256'].items():
        if file_hash(math_data/name)!=digest:raise ValueError('Frozen original math data changed')
    selected_visual_base=base
    code_manifest=json.loads((code/'corpus.json').read_text())
    for split in code_manifest['splits'].values():
        for shard in split['shards']:
            if file_hash(code/shard['path'])!=shard['sha256']:raise ValueError('Expanded code token shard changed')
    try:
        base,warmup_plan=foundation_warmup(base,code,wiki,reports,status)
    except Exception as error:
        status('requires inspection; foundation warmup stopped',error=str(error));raise
    while not (wiki/'corpus.json').exists():
        status('waiting for remaining Wikipedia corpus after completed foundation warmup',
               pending=[str(wiki/'corpus.json')],completed_warmup_steps=warmup_plan['steps']);time.sleep(30)
    for root in [code,wiki]:
        manifest=json.loads((root/'corpus.json').read_text())
        for split in manifest['splits'].values():
            for shard in split['shards']:
                if file_hash(root/shard['path'])!=shard['sha256']:raise ValueError('Expanded token shard changed')
        if root==code:
            for split,digest in manifest['sft_files_sha256'].items():
                if file_hash(root/('sft.'+split+'.jsonl'))!=digest:raise ValueError('Expanded code instruction file changed')
    curriculum=ROOT/'data/extended-curriculum-v1';curriculum.mkdir(exist_ok=True)
    packed=curriculum/'packed'
    if not (packed/'corpus.json').exists():mixture=mix([('documented-code',code,None),('encyclopedia',wiki,None)],packed)
    else:mixture=json.loads((packed/'corpus.json').read_text())
    # Original code/document pairs already exclude exact reserved questions.
    # Retain No Robots demonstrations from the earlier audited instruction file.
    old=ROOT/'data/alignment-curriculum-v1';source_instructions=[]
    for row in jsonl(old/'instruction.train.jsonl'):
        if row.get('source')=='HuggingFaceH4/no_robots':source_instructions.append(row)
    count=0;balanced=defaultdict(list)
    with (curriculum/'code-instruction.train.jsonl').open('w') as target:
        for row in source_instructions:target.write(json.dumps(row,ensure_ascii=True)+'\n');count+=1
        for row in jsonl(code/'sft.train.jsonl'):
            target.write(json.dumps(row,ensure_ascii=True)+'\n');count+=1
            bucket=balanced[row['language']]
            if len(bucket)<3000:bucket.append(row)
    verified_repetitions=3;verified_records=0
    with (curriculum/'verified-code.train.jsonl').open('w') as target:
        for row in source_instructions:target.write(json.dumps(row,ensure_ascii=True)+'\n');verified_records+=1
        # Keep a multilingual code sample while emphasizing standalone programs
        # with published behavior checks. Repetitions are recorded as exposures.
        for bucket in balanced.values():
            for row in bucket[:600]:target.write(json.dumps(row,ensure_ascii=True)+'\n');verified_records+=1
        for _ in range(verified_repetitions):
            with (verified/'train.jsonl').open() as source:
                for line in source:target.write(line);verified_records+=1
    math_rows=list(jsonl(math_data/'train.jsonl'));math_records=0
    with (curriculum/'math-reasoning.train.jsonl').open('w') as target:
        for row in [*math_rows,*source_instructions]:target.write(json.dumps(row,ensure_ascii=True)+'\n');math_records+=1
        for bucket in balanced.values():
            for row in bucket[:600]:target.write(json.dumps(row,ensure_ascii=True)+'\n');math_records+=1
        with (verified/'train.jsonl').open() as source:
            for line in source:target.write(line);math_records+=1
    # Preserve a bounded lettering supplement alongside all original real photos
    # and reading tasks, after the much longer language-only stages.
    rendered=ROOT/'data/rendered-text-v1'
    rendered_manifest=json.loads((rendered/'manifest.json').read_text())
    if file_hash(rendered/'train.jsonl')!=rendered_manifest['files_sha256']['train.jsonl']:raise ValueError('Rendered instruction records changed')
    available=rendered_manifest['splits']['train']['examples'];cap=min(10000,available);retained=0
    with (curriculum/'images.train.jsonl').open('w') as target:
        with (old/'images.train.jsonl').open() as source:
            for line in source:target.write(line)
        for i,row in enumerate(jsonl(rendered/'train.jsonl')):
            if (i+1)*cap//available==i*cap//available:continue
            row['image']=str((rendered/row['image']).resolve())
            row['font_notice']=str((rendered/row['font_notice']).resolve())
            target.write(json.dumps(row,ensure_ascii=True)+'\n');retained+=1
        with (objects/'train.jsonl').open() as source:
            for line in source:target.write(line)
    if retained!=cap:raise ValueError('Rendered retention count differs from source manifest')
    joint_rows=source_instructions+[r for bucket in balanced.values() for r in bucket]
    with (curriculum/'joint.train.jsonl').open('w') as target:
        for row in joint_rows:target.write(json.dumps(row,ensure_ascii=True)+'\n')
        for row in math_rows:target.write(json.dumps(row,ensure_ascii=True)+'\n')
        with (curriculum/'images.train.jsonl').open() as source:
            for line in source:target.write(line)
        with (verified/'train.jsonl').open() as source:
            for line in source:target.write(line)
    images=sum(1 for _ in jsonl(curriculum/'images.train.jsonl'));captions=sum(1 for _ in jsonl(old/'contrastive.train.jsonl'))
    specs=[
        {'name':'language-pretraining','stage':'pretrain','data':packed,'trainable':'language','batch':1,'accumulation':16,
         'steps':2*((mixture['splits']['train']['tokens']-1)//262080),'lr':1e-4},
        {'name':'multilingual-code-instruction','stage':'sft','data':curriculum/'code-instruction.train.jsonl','trainable':'language',
         'batch':2,'accumulation':4,'steps':math.ceil(count/32),'lr':2e-5},
        {'name':'verified-standalone-code','stage':'sft','data':curriculum/'verified-code.train.jsonl','trainable':'language',
         'batch':2,'accumulation':4,'steps':math.ceil(2*verified_records/32),'lr':1e-5},
        {'name':'math-reasoning','stage':'sft','data':curriculum/'math-reasoning.train.jsonl','trainable':'language',
         'batch':2,'accumulation':4,'steps':math.ceil(2*math_records/32),'lr':1e-5},
        {'name':'vision-realignment','stage':'contrastive','data':old/'contrastive.train.jsonl','trainable':'vision',
         'batch':8,'accumulation':1,'steps':math.ceil(captions/32),'lr':5e-5},
        {'name':'image-reading','stage':'sft','data':curriculum/'images.train.jsonl','trainable':'vision',
         'batch':2,'accumulation':2,'steps':math.ceil(images/16),'lr':5e-5},
        {'name':'joint','stage':'sft','data':curriculum/'joint.train.jsonl','trainable':'all',
         'batch':2,'accumulation':2,'steps':math.ceil((images+len(joint_rows)+len(math_rows)+verified_manifest['train_examples'])/16),'lr':1e-5},
    ]
    for spec in specs:
        if spec['steps']<1:raise ValueError('Expanded training budget is empty')
        spec['image_size']=448
        spec['data_sha256']=file_hash(spec['data']/'corpus.json' if spec['data'].is_dir() else spec['data'])
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'base_checkpoint':str(base),
          'selected_visual_checkpoint_before_foundation_warmup':str(selected_visual_base),
          'foundation_warmup_plan_sha256':file_hash(reports/'foundation-warmup-plan.json'),
          'foundation_warmup_target_token_exposures':warmup_plan['planned_target_tokens'],
          'lettering_selection_sha256':file_hash(lettering/'selection.json'),'rendered_reading_examples_retained':retained,
          'object_question_manifest_sha256':file_hash(objects/'manifest.json'),'object_training_examples':object_manifest['splits']['train']['examples'],
          'verified_code_manifest_sha256':file_hash(verified/'manifest.json'),
          'verified_unique_programs':verified_manifest['unique_programs'],'verified_code_name_variant_rows':verified_manifest['train_examples'],
          'verified_code_row_repetitions':verified_repetitions,'verified_stage_passes':2,
          'public_code_assessment_manifest_sha256':file_hash(public_code/'manifest.json'),
          'cpp_reference_readiness_sha256':file_hash(native_readiness),
          'math_manifest_sha256':file_hash(math_data/'manifest.json'),
          'math_unique_training_problems':len(math_rows),'math_stage_passes':2,
          'official_math_test_problems':math_manifest['official_test_examples'],
          'unique_pretraining_tokens':mixture['splits']['train']['tokens'],
          'pretraining_passes':2,'planned_pretraining_target_tokens':specs[0]['steps']*262080,
          'stages':[{**s,'data':str(s['data'])} for s in specs],
          'goal_achieved':False,'limitations':['Repeated passes are training exposures, not additional unique data.',
             'Public source examples contain dependencies and source-era APIs; correctness/generalization still needs testing.',
             'More training and green pipeline checks do not establish high-end-model parity.']}
    atomic_json(reports/'plan.json',plan);checkpoint=base;previous=base.parent
    env={**os.environ,'OMP_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
    try:
        # Compare the admitted policy before any new training on this QA set.
        baseline=reports/'object-baseline.json'
        if baseline.exists():
            prior_object=json.loads(baseline.read_text())
            if prior_object['checkpoint_sha256']!=file_hash(base) or prior_object['data_sha256']!=file_hash(objects/'valid.jsonl'):raise ValueError('Object baseline differs from the admitted policy/data')
        else:
            status('evaluating object-question baseline before extended training')
            subprocess.run([sys.executable,'-u','-m','slm_lab.object_evaluation','--checkpoint',str(base),
                            '--data',str(objects/'valid.jsonl'),'--output',str(baseline),'--device','cuda'],check=True,cwd=ROOT,env=env)
        for spec in specs:
            directory=ROOT/'checkpoints/extended-curriculum-v1'/spec['name']
            if not complete_stage(directory,spec):
                if directory.exists() and any(directory.iterdir()):raise ValueError('Incomplete extended stage needs inspection/resume')
                if shutil.disk_usage(ROOT).free<25*1024**3:raise ValueError('Insufficient disk for atomic extended-training checkpoints')
                command=[str(ROOT/'.venv/bin/torchrun'),'--standalone','--nproc_per_node=4','-m','slm_lab.training',spec['stage'],
                    '--checkpoint',str(checkpoint),'--tokenizer',str(checkpoint.parent/'tokenizer.json'),
                    '--data',str(spec['data']),'--output',str(directory),'--trainable',spec['trainable'],
                    '--steps',str(spec['steps']),'--batch-size',str(spec['batch']),'--gradient-accumulation',str(spec['accumulation']),
                    '--learning-rate',str(spec['lr']),'--image-size','448','--device','cuda','--threads','1',
                    '--gradient-checkpointing','--checkpoint-every','1000','--keep-checkpoints','1','--log-every','10']
                status('training extended public-data curriculum',stage=spec['name'],planned_steps=spec['steps'])
                subprocess.run(command,check=True,cwd=ROOT,env=env)
                if not complete_stage(directory,spec):raise ValueError('Extended stage did not finish its budget')
            checkpoint=directory/'model.pt'
            if previous:
                for path in [previous/'training-state.pt',*previous.glob('step-*.pt')]:path.unlink(missing_ok=True)
            previous=directory
        for path in previous.glob('step-*.pt'):path.unlink(missing_ok=True)
        for module,name,extra in [
            ('slm_lab.code_quality','code-quality.json',[]),
            ('slm_lab.knowledge_evaluation','knowledge.json',['--data',str(ROOT/'data/knowledge-eval-v1/questions.jsonl')]),
            ('slm_lab.math_evaluation','math.json',['--data',str(math_data/'test.jsonl')]),
            ('slm_lab.vision_evaluation','vision.json',['--data',str(old/'contrastive.valid.jsonl')]),
            ('slm_lab.ocr_evaluation','ocr.json',['--data',str(old/'images.valid.jsonl')]),
            ('slm_lab.object_evaluation','objects.json',['--data',str(objects/'valid.jsonl')])]:
            status('evaluating completed extended curriculum',task=module)
            subprocess.run([sys.executable,'-u','-m',module,'--checkpoint',str(checkpoint),'--device','cuda',
                            '--output',str(reports/name),*extra],check=True,cwd=ROOT,env=env)
        status('evaluating supported public-programming benchmark')
        subprocess.run([sys.executable,'-u','-m','slm_lab.public_code_benchmark','--checkpoint',str(checkpoint),
                        '--data',str(public_code/'questions.jsonl'),'--output',str(reports/'public-code.json'),
                        '--device','cuda'],check=True,cwd=ROOT,env=env)
        status('evaluating completed policy on native C++ code')
        subprocess.run([sys.executable,'-u','-m','slm_lab.cpp_evaluation','--checkpoint',str(checkpoint),
                        '--output',str(reports/'cpp-code.json'),'--device','cuda'],check=True,cwd=ROOT,env=env)
        atomic_json(reports/'summary.json',{'checkpoint':str(checkpoint),'checkpoint_sha256':file_hash(checkpoint),
                    'goal_achieved':False,'high_end_llm_parity':'Not established',
                    'pending':['Fresh human-feedback training and policy comparisons on this expanded model',
                               'Broader languages/repository-level tasks','Matched high-end-model capability comparisons']})
        status('extended language and multimodal training/evaluation completed; broader capability unverified',checkpoint=str(checkpoint))
    except Exception as error:
        status('requires inspection; extended pipeline stopped',error=str(error));raise

if __name__=='__main__':
    if '--resume-frozen-plan' in sys.argv:
        from slm_lab.extended_curriculum_recovery import main as resume_frozen_plan
        resume_frozen_plan()
    else:
        main()
