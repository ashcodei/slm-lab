"""Own-vision character-sequence experiment with fixed own-language prototypes."""
import argparse
from collections import Counter,defaultdict
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image,ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer

from .corpus import file_hash
from .diverse_images import atomic_json,jsonl
from .model import ModelConfig,VisionEncoder
from .ocr_evaluation import transcription_scores
from .rendered_text import text_group

ALPHABET=''.join(chr(i) for i in range(32,127))


def labels(text):
    if not text or any(char not in ALPHABET for char in text):raise ValueError('Unsupported nonempty printable-ASCII transcript')
    return [ALPHABET.index(char)+1 for char in text]


def minimum_frames(text):
    return len(labels(text))+sum(a==b for a,b in zip(text,text[1:]))


def collapse(path):
    result=[];previous=None
    for value in path:
        value=int(value)
        if not 0<=value<=len(ALPHABET):raise ValueError('Invalid CTC path class')
        if value and value!=previous:result.append(ALPHABET[value-1])
        previous=value
    return ''.join(result)


def prepare(rendered,real,output,frames=56,size=448):
    rendered,real,output=map(Path,[rendered,real,output]);output.mkdir(parents=True,exist_ok=True)
    sources=[]
    for root,names in [(rendered,['train.jsonl','valid.jsonl']),(real,['images.train.jsonl','images.valid.jsonl'])]:
        manifest=json.loads((root/'manifest.json').read_text())
        for name in names:
            if file_hash(root/name)!=manifest['files_sha256'][name]:raise ValueError('Audited reading source changed')
        sources.append({'root':str(root.resolve()),'manifest_sha256':file_hash(root/'manifest.json')})
    config={'version':1,'frames':frames,'image_size':size,'alphabet':ALPHABET,'sources':sources}
    path=output/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        if manifest['configuration']!=config:raise ValueError('Use a new character-sequence data directory')
        for name,digest in manifest['files_sha256'].items():
            if file_hash(output/name)!=digest:raise ValueError('Frozen character-sequence records changed')
        return manifest
    protected={text_group(row['response']) for row in jsonl(rendered/'valid.jsonl')}
    rows={'train':[],'valid':[]};excluded=[];groups={split:set() for split in rows};training_text=set()
    for root,names,family in [(rendered,{'train':'train.jsonl','valid':'valid.jsonl'},'rendered'),
                              (real,{'train':'images.train.jsonl','valid':'images.valid.jsonl'},'real_word')]:
        for split,name in names.items():
            for row in jsonl(root/name):
                if row.get('task')!='word_ocr':continue
                try:
                    if minimum_frames(row['response'])>frames:raise ValueError('Transcript needs more CTC frames')
                    if split=='train' and text_group(row['response']) in protected:raise ValueError('Protected rendered-validation text group')
                except ValueError as error:
                    excluded.append({'id':row['id'],'split':split,'family':family,'reason':str(error)});continue
                row={**row,'image':str((root/row['image']).resolve()),'ctc_family':family}
                if row.get('font_notice'):row['font_notice']=str((root/row['font_notice']).resolve())
                rows[split].append(row);groups[split].add(row['group_id'])
                if split=='train':training_text.add(text_group(row['response']))
    if groups['train']&groups['valid']:raise ValueError('Reading parent groups overlap')
    if protected&training_text:raise ValueError('Rendered validation words entered sequence training')
    for name,records in [*rows.items(),('excluded',excluded)]:
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(row,ensure_ascii=True)+'\n' for row in records))
    manifest={'configuration':config,'splits':{name:{'examples':len(records),'by_family':dict(Counter(row['ctc_family'] for row in records))} for name,records in rows.items()},
              'excluded':dict(Counter(row['reason'] for row in excluded)),
              'parent_group_overlap':0,'rendered_validation_text_group_overlap_with_training':0,
              'files_sha256':{name:file_hash(output/name) for name in ['train.jsonl','valid.jsonl','excluded.jsonl']},
              'limitations':['Existing development splits, not independent final assessment.',
                             'Real word vocabulary may occur on both image splits; rendered validation text groups are protected for this objective.',
                             'Printable ASCII word crops only; whole pages, handwriting and multilingual scripts are not trained by this loss.']}
    atomic_json(path,manifest);return manifest


class CharacterSequenceModel(nn.Module):
    def __init__(self,vision,prototypes,frames=56,temperature=16.):
        super().__init__();self.vision=vision;self.frames=frames;self.temperature=temperature
        if prototypes.shape!=(len(ALPHABET)+1,vision.config.hidden_size):raise ValueError('Character prototype shape differs from own language space')
        if frames<1 or temperature<=0 or not torch.isfinite(prototypes).all():raise ValueError('Invalid character-sequence configuration')
        self.register_buffer('prototypes',F.normalize(prototypes.detach().float().clone(),dim=-1))

    def forward(self,pixels):
        saved=[]
        handle=self.vision.norm.register_forward_hook(lambda module,args,out:saved.append(out))
        try:self.vision(pixels,gradient_checkpointing=self.training)
        finally:handle.remove()
        if len(saved)!=1:raise ValueError('Expected one unpooled own-vision feature grid')
        values=saved[0];side=math.isqrt(values.shape[1])
        if side*side!=values.shape[1]:raise ValueError('Unpooled features are not a square grid')
        columns=values.reshape(len(values),side,side,values.shape[-1]).mean(1)
        projected=self.vision.project(columns)
        # Existing projector and encoder are trained; character prototypes remain
        # fixed buffers. FP32 similarity/log-softmax/CTC avoids BF16 loss rounding.
        with torch.autocast(device_type=pixels.device.type,enabled=False):
            sequence=F.interpolate(projected.float().transpose(1,2),size=self.frames,mode='linear',align_corners=False).transpose(1,2)
            return self.temperature*(F.normalize(sequence,dim=-1)@self.prototypes.float().T)


def ctc_loss(logits,texts):
    targets=[labels(text) for text in texts]
    if any(minimum_frames(text)>logits.shape[1] for text in texts):raise ValueError('Unalignable CTC transcript')
    if len(targets)!=len(logits):raise ValueError('Transcript batch differs from logits')
    packed=torch.tensor([value for target in targets for value in target],device=logits.device,dtype=torch.long)
    lengths=torch.tensor([len(target) for target in targets],dtype=torch.long)
    input_lengths=torch.full((len(logits),),logits.shape[1],dtype=torch.long)
    loss=F.ctc_loss(logits.float().log_softmax(-1).transpose(0,1),packed,input_lengths,lengths,
                    blank=0,reduction='mean',zero_infinity=False)
    if not torch.isfinite(loss):raise ValueError('Non-finite CTC loss; no zero-infinity masking')
    return loss


def from_base(checkpoint,frames=56):
    obj=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True);config=ModelConfig(**obj['config'])
    tokenizer_path=Path(checkpoint).parent/'tokenizer.json'
    if file_hash(tokenizer_path)!=obj['tokenizer_sha256']:raise ValueError('Own tokenizer changed')
    tokenizer=Tokenizer.from_file(str(tokenizer_path));ids=[tokenizer.token_to_id('<pad>')]
    for char in ALPHABET:
        encoded=tokenizer.encode(char).ids
        if len(encoded)!=1:raise ValueError('Own tokenizer character needs multiple entries')
        ids.append(encoded[0])
    if None in ids or len(set(ids))!=len(ids):raise ValueError('Character/blank prototype IDs collide')
    vision=VisionEncoder(config);vision.load_state_dict({name.removeprefix('vision.'):value for name,value in obj['model'].items() if name.startswith('vision.')})
    prototypes=obj['model']['embedding.weight'][ids].clone()
    model=CharacterSequenceModel(vision,prototypes,frames)
    return model,{'config':asdict(config),'base_checkpoint':str(Path(checkpoint).resolve()),'base_checkpoint_sha256':file_hash(checkpoint),
                  'tokenizer_sha256':obj['tokenizer_sha256'],'prototype_token_ids':ids,'frames':frames,'temperature':16.}


def batch_pixels(rows,size,device):
    result=[]
    for row in rows:
        path=Path(row['image'])
        if file_hash(path)!=row['stored_image_sha256']:raise ValueError('Reading image changed')
        with Image.open(path) as image:
            image=ImageOps.pad(image.convert('RGB'),(size,size),method=Image.Resampling.BICUBIC,color=(128,128,128))
            values=np.asarray(image,dtype=np.float32)/127.5-1
            result.append(torch.from_numpy(values.copy()).permute(2,0,1))
    return torch.stack(result).to(device)


def atomic_torch(path,value):
    temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)


@torch.no_grad()
def evaluate(model,rows,output,device,size,maximum_per_family=128):
    model.eval();selected=[]
    for family in sorted({row['ctc_family'] for row in rows}):
        used=set()
        for row in sorted([row for row in rows if row['ctc_family']==family],key=lambda r:hashlib.sha256(r['id'].encode()).digest()):
            if row['group_id'] in used:continue
            selected.append(row);used.add(row['group_id'])
            if len(used)>=maximum_per_family:break
    amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    results=[]
    for i,row in enumerate(selected):
        other=next(candidate for candidate in selected[i+1:]+selected[:i] if candidate['ctc_family']==row['ctc_family'] and candidate['group_id']!=row['group_id'] and candidate['response']!=row['response'])
        pixels=batch_pixels([row,other],size,device)
        with amp():logits=model(torch.cat([pixels[:1],torch.zeros_like(pixels[:1]),pixels[1:]],0))
        predictions=[collapse(path) for path in logits.argmax(-1).tolist()]
        result={'id':row['id'],'group_id':row['group_id'],'family':row['ctc_family'],'expected':row['response'],
                'predicted':predictions[0],'blank_predicted':predictions[1],'wrong_predicted':predictions[2],
                **transcription_scores(row['response'],predictions[0])}
        results.append(result)
    def summarize(items):
        chars=sum(item['reference_characters'] for item in items)
        return {'examples':len(items),'exact_accuracy':sum(item['exact_match'] for item in items)/len(items),
                'character_error_rate':sum(item['character_edits'] for item in items)/chars,
                'blank_original_target_match':sum(item['blank_predicted']==item['expected'] for item in items)/len(items),
                'wrong_original_target_match':sum(item['wrong_predicted']==item['expected'] for item in items)/len(items)}
    report={'overall':summarize(results),'by_family':{family:summarize([row for row in results if row['family']==family]) for family in sorted({r['family'] for r in results})},
            'frames':model.frames,'image_size':size,'protocol':'Fixed greedy CTC path collapse, '+getattr(model,'readout_kind','own fixed character prototypes')+', unique-parent samples with blank/wrong-image controls; no language decoder or repair.',
            'results':results,'goal_achieved':False,'limitations':['Component reading diagnostic, not the production language model\u2019s native transcription.','Existing tuning splits; not broad or independently certified OCR.']}
    atomic_json(output,report);return report


def train(checkpoint,data,output,reports,device='cuda:3',passes=1,batch_size=8,accumulation=4,learning_rate=1e-4,resume=False,
          model_factory=None,head_warmup_steps=0,encoder_rate_scale=1.):
    torch.set_num_threads(1);torch.manual_seed(42)
    if device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.25,int(device.split(':')[1]) if ':' in device else 0)
    data,output,reports=map(Path,[data,output,reports]);output.mkdir(parents=True,exist_ok=True);reports.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((data/'manifest.json').read_text())
    for name,digest in manifest['files_sha256'].items():
        if file_hash(data/name)!=digest:raise ValueError('Frozen sequence curriculum changed')
    rows=list(jsonl(data/'train.jsonl'));valid=list(jsonl(data/'valid.jsonl'))
    model,meta=(model_factory or from_base)(checkpoint,manifest['configuration']['frames']);model=model.to(device)
    steps=math.ceil(passes*len(rows)/(batch_size*accumulation));size=manifest['configuration']['image_size']
    plan={**meta,'manifest_sha256':file_hash(data/'manifest.json'),'train_rows':len(rows),'passes':passes,'steps':steps,
          'batch_size':batch_size,'accumulation':accumulation,'learning_rate':learning_rate,'image_size':size,
          'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
          'new_trainable_parameters':meta.get('new_trainable_parameters',0),
          'head_warmup_steps':head_warmup_steps,'encoder_rate_scale':encoder_rate_scale,
          'implementation_sha256':file_hash(__file__),'goal_achieved':False,
          'scope':'Own vision/projector sequence learning only; native multimodal generation requires separate adaptation and matched evaluation.'}
    path=reports/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Sequence experiment differs from its recorded plan')
    if any(output.iterdir()) and not resume:raise ValueError('Existing sequence checkpoints require explicit inspected resume')
    atomic_json(path,plan)
    encoder=[p for name,p in model.named_parameters() if name.startswith('vision.') and p.requires_grad]
    head=[p for name,p in model.named_parameters() if not name.startswith('vision.') and p.requires_grad]
    optimizer=torch.optim.AdamW([{'params':encoder,'rate_scale':encoder_rate_scale},
                                {'params':head,'rate_scale':1.}],lr=learning_rate,weight_decay=.01)
    start=0
    if resume:
        state=torch.load(output/'latest.pt',map_location=device,weights_only=True)
        if state['plan']!=plan:raise ValueError('Resume sequence plan differs')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start=state['steps']
        torch.set_rng_state(state['torch_rng_state'].cpu())
        if device.startswith('cuda'):torch.cuda.set_rng_state(state['cuda_rng_state'].cpu(),device)
    else:
        atomic_json(reports/'status.json',{'status':'evaluating initial own-character readout'})
        evaluate(model,valid,reports/'baseline.json',device,size)
    # Finite deterministic sample order; repeated passes are explicitly exposures.
    generator=torch.Generator().manual_seed(42);order=[]
    for _ in range(passes):order.extend(torch.randperm(len(rows),generator=generator).tolist())
    total=steps*batch_size*accumulation
    order=(order*((total+len(order)-1)//len(order)))[:total]
    amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    started=time.monotonic()
    def save(name,completed,optimizer_state=True):
        value={'model':{name:tensor.detach().cpu() for name,tensor in model.state_dict().items()},'plan':plan,'steps':completed,
               'torch_rng_state':torch.get_rng_state()}
        if optimizer_state:value['optimizer']=optimizer.state_dict()
        if device.startswith('cuda'):value['cuda_rng_state']=torch.cuda.get_rng_state(device)
        atomic_torch(output/name,value)
    for step in range(start,steps):
        if step==start or step==head_warmup_steps:
            model.vision.requires_grad_(step>=head_warmup_steps)
            if getattr(model,'freeze_projector',False):model.vision.project.requires_grad_(False)
        model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0.;begin=time.monotonic()
        lr=learning_rate*min(1,(step+1)/max(1,steps//10))*(.1+.9*(1+math.cos(math.pi*step/steps))/2)
        for group in optimizer.param_groups:group['lr']=lr*group['rate_scale']
        for micro in range(accumulation):
            offset=(step*accumulation+micro)*batch_size;batch=[rows[index] for index in order[offset:offset+batch_size]]
            pixels=batch_pixels(batch,size,device)
            with amp():logits=model(pixels)
            loss=ctc_loss(logits,[row['response'] for row in batch]);(loss/accumulation).backward();loss_sum+=float(loss.detach())/accumulation
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step==0 or (step+1)%10==0 or step+1==steps:
            progress={'status':'training own-vision character sequences','step':step+1,'planned_steps':steps,'loss':loss_sum,
                      'gradient_norm':float(norm),'learning_rate':lr,'step_seconds':time.monotonic()-begin,
                      'elapsed_seconds':time.monotonic()-started,'sample_exposures':(step+1)*batch_size*accumulation}
            atomic_json(reports/'status.json',progress);print(json.dumps(progress),flush=True)
        if (step+1)%100==0:save('latest.pt',step+1)
    save('model.pt',steps,False);save('latest.pt',steps)
    report=evaluate(model,valid,reports/'final-reading.json',device,size)
    atomic_json(reports/'status.json',{'status':'character-sequence component training/evaluation completed; native language reading unverified',
                                     'steps':steps,'checkpoint':str(output/'model.pt'),'component_exact_accuracy':report['overall']['exact_accuracy'],'goal_achieved':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--rendered',default='data/rendered-text-v1');parser.add_argument('--real',default='data/alignment-curriculum-v1')
    parser.add_argument('--data',required=True);parser.add_argument('--checkpoint');parser.add_argument('--output');parser.add_argument('--reports')
    parser.add_argument('--device',default='cuda:3');parser.add_argument('--passes',type=int,default=1);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.prepare:print(json.dumps(prepare(args.rendered,args.real,args.data),indent=2))
    else:
        if not args.checkpoint or not args.output or not args.reports or args.passes<1:parser.error('Training needs base checkpoint, outputs and positive passes')
        try:train(args.checkpoint,args.data,args.output,args.reports,args.device,args.passes,resume=args.resume)
        except Exception as error:
            Path(args.reports).mkdir(parents=True,exist_ok=True)
            atomic_json(Path(args.reports)/'status.json',{'status':'requires inspection; character-sequence process stopped','error':str(error),'goal_achieved':False})
            raise


if __name__=='__main__':main()
