"""Frozen own-encoder glyph diagnostics; never a production OCR capability claim."""
import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F

from .corpus import file_hash
from .diverse_images import atomic_json,jsonl
from .evaluation_provenance import capture,seal
from .model import ModelConfig,TinyMultimodalLM,VisionEncoder
from .rendered_text import ALPHABET,HELD_OUT_FONTS,render,pixel_hash


def prepare(output,fonts,reference_manifest,notice,train_per_class=28,valid_per_class=12,seed=9107,size=448):
    output,fonts=Path(output),Path(fonts);output.mkdir(parents=True,exist_ok=True)
    if min(train_per_class,valid_per_class)<2:raise ValueError('Need repeated renders in each split')
    original=json.loads(Path(reference_manifest).read_text())
    expected=original['configuration']['fonts']
    paths=sorted(fonts.glob('Liberation*.ttf'))
    if {p.name:file_hash(p) for p in paths}!=expected:raise ValueError('Font hashes differ from the audited renderer sources')
    if file_hash(notice)!=original['configuration']['font_license_sha256']:raise ValueError('Font notice differs')
    configuration={'version':1,'seed':seed,'image_size':size,'alphabet':ALPHABET,
                   'train_per_class':train_per_class,'valid_per_class':valid_per_class,
                   'font_hashes':expected,'notice_sha256':file_hash(notice),
                   'renderer_sha256':file_hash(Path(__file__).with_name('rendered_text.py'))}
    path=output/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        if manifest['configuration']!=configuration:raise ValueError('Use a new glyph diagnostic directory')
        for name,digest in manifest['files_sha256'].items():
            if file_hash(output/name)!=digest:raise ValueError('Glyph diagnostic records changed')
        return manifest
    (output/'images').mkdir(exist_ok=True);(output/'font-license.txt').write_bytes(Path(notice).read_bytes())
    groups={'train':[p for p in paths if p.name not in HELD_OUT_FONTS],
            'valid':[p for p in paths if p.name in HELD_OUT_FONTS]}
    pixels={'train':set(),'valid':set()};records={'train':[],'valid':[]}
    for split,count in [('train',train_per_class),('valid',valid_per_class)]:
        for label,character in enumerate(ALPHABET):
            for index in range(count):
                font=groups[split][index%len(groups[split])]
                trial=seed+label*10000+index*100+(0 if split=='train' else 1_000_000)
                for attempt in range(100):
                    image,metadata=render(character,font,trial+attempt);digest=pixel_hash(image,size)
                    if digest not in pixels['train'] and digest not in pixels['valid']:break
                else:raise ValueError('Unable to produce a distinct model-input glyph')
                pixels[split].add(digest);name=f'{split}-{label:02d}-{index:03d}'
                image_path=output/'images'/(name+'.png');image.save(image_path,compress_level=1)
                records[split].append({'id':name,'character':character,'label':label,'image':str(image_path.relative_to(output)),
                                       'font_name':font.name,'font_sha256':expected[font.name],
                                       'render_seed':trial+attempt,'rendering':metadata,
                                       'stored_image_sha256':file_hash(image_path),'model_pixel_sha256':digest})
    for split,rows in records.items():(output/(split+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    manifest={'configuration':configuration,'splits':{name:{'examples':len(rows),'class_counts':dict(Counter(r['character'] for r in rows))} for name,rows in records.items()},
              'train_valid_exact_input_overlap':len(pixels['train']&pixels['valid']),
              'held_out_font_styles':sorted(HELD_OUT_FONTS),
              'files_sha256':{name:file_hash(output/name) for name in ['train.jsonl','valid.jsonl','font-license.txt']},
              'scope':'Synthetic single-glyph diagnostics, used only for frozen-feature classifiers; no encoder/language/production-weight updates.',
              'limitations':['All character classes occur in both splits; font styles and rendering seeds are held out.',
                             'This is not word/scene/document recognition or an independent broad vision benchmark.',
                             'A trained probe head is never deployed or transferred to the production model.']}
    atomic_json(path,manifest);return manifest


def fit_ridge(features,labels,classes,regularization=1.):
    if features.ndim!=2 or len(features)!=len(labels) or len(features)<2 or regularization<=0:
        raise ValueError('Invalid ridge training inputs')
    x=features.double();mean=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(1e-6)
    x=(x-mean)/scale
    y=F.one_hot(labels.long(),classes).double();intercept=y.mean(0);y=y-intercept
    n,dim=x.shape
    if not torch.isfinite(x).all():raise ValueError('Non-finite glyph features')
    if dim<=n:
        matrix=x.T@x/n+regularization*torch.eye(dim,dtype=torch.float64)
        coefficients=torch.linalg.solve(matrix,x.T@y/n)
    else:
        matrix=x@x.T/n+regularization*torch.eye(n,dtype=torch.float64)
        coefficients=x.T@torch.linalg.solve(matrix,y)/n
    return {'mean':mean,'scale':scale,'coefficients':coefficients,'intercept':intercept,
            'regularization':regularization}


def scores(head,features):
    return (features.double()-head['mean'])/head['scale']@head['coefficients']+head['intercept']


def state_digest(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def features(model,pixels,grid=2):
    saved=[]
    def hook(module,args):
        values=args[0];side=int(values.shape[1]**.5)
        if side*side!=values.shape[1]:raise ValueError('Vision prefix is not a spatial grid')
        values=values.transpose(1,2).reshape(len(values),values.shape[2],side,side)
        saved.append(F.adaptive_avg_pool2d(values,(grid,grid)).flatten(1).float().cpu())
    handle=model.project.register_forward_pre_hook(hook)
    try:model(pixels)
    finally:handle.remove()
    if len(saved)!=1:raise ValueError('Expected exactly one vision projection')
    return saved[0]


def load_vision(checkpoint,config=None,seed=42):
    torch.manual_seed(seed)
    if checkpoint is not None:
        obj=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        current=ModelConfig(**obj['config'])
        if config is not None and current!=config:raise ValueError('Probe encoder architectures differ')
        model=VisionEncoder(current)
        model.load_state_dict({key.removeprefix('vision.'):value for key,value in obj['model'].items() if key.startswith('vision.')})
    else:
        if config is None:raise ValueError('Random control needs the matched architecture')
        current=config;model=VisionEncoder(current);model.apply(TinyMultimodalLM._initialize)
        torch.nn.init.normal_(model.position,std=.02)
    model.eval().requires_grad_(False);return model,current


def extract(model,rows,root,size,device,batch_size=16):
    result=[];amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    for start in range(0,len(rows),batch_size):
        images=[]
        for row in rows[start:start+batch_size]:
            path=root/row['image']
            if file_hash(path)!=row['stored_image_sha256']:raise ValueError('Diagnostic glyph image changed')
            with Image.open(path) as image:
                image=ImageOps.pad(image.convert('RGB'),(size,size),method=Image.Resampling.BICUBIC,color=(128,128,128))
                if hashlib.sha256(image.tobytes()).hexdigest()!=row['model_pixel_sha256']:raise ValueError('Glyph input transform changed')
                array=np.asarray(image,dtype=np.float32)/127.5-1
                images.append(torch.from_numpy(array.copy()).permute(2,0,1))
        with amp():result.append(features(model,torch.stack(images).to(device)))
    return torch.cat(result)


def evaluate(checkpoints,data,output,device='cuda:3',random_control=True,regularization=1.):
    torch.set_num_threads(1)
    if device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.15,int(device.split(':')[1]) if ':' in device else 0)
    provenance=capture();implementation=file_hash(__file__)
    data,output=Path(data),Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((data/'manifest.json').read_text())
    for name,digest in manifest['files_sha256'].items():
        if file_hash(data/name)!=digest:raise ValueError('Frozen glyph records changed')
    train=list(jsonl(data/'train.jsonl'));valid=list(jsonl(data/'valid.jsonl'))
    labels=torch.tensor([row['label'] for row in train]);targets=torch.tensor([row['label'] for row in valid])
    config=None;results=[]
    entries=list(checkpoints)
    # Load a trained architecture before constructing the matching random control.
    if random_control:entries.append(('random',None))
    for name,checkpoint in entries:
        model,config=load_vision(checkpoint,config);model=model.to(device)
        before=state_digest(model);size=manifest['configuration']['image_size'];config.image_token_count(size)
        print('extracting frozen glyph features',name,flush=True)
        training=extract(model,train,data,size,device);validation=extract(model,valid,data,size,device)
        amp=torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
        with amp:blank=features(model,torch.zeros(1,3,size,size,device=device))
        if state_digest(model)!=before:raise ValueError('Frozen vision weights changed during the probe')
        del model
        if device.startswith('cuda'):torch.cuda.empty_cache()
        print('fitting diagnostic ridge classifier',name,flush=True)
        head=fit_ridge(training,labels,len(ALPHABET),regularization)
        train_predictions=scores(head,training).argmax(1);predictions=scores(head,validation).argmax(1)
        blank_predictions=scores(head,blank).argmax(1).repeat(len(valid))
        # Shift by one full class block, ensuring every wrong image has a different glyph label.
        shift=manifest['configuration']['valid_per_class'];wrong=validation.roll(-shift,0)
        if torch.any(targets.roll(-shift,0)==targets):raise ValueError('Wrong-image controls share glyph labels')
        wrong_predictions=scores(head,wrong).argmax(1)
        rows=[{'id':row['id'],'expected':row['character'],'predicted':ALPHABET[int(pred)],
               'blank_predicted':ALPHABET[int(blank_pred)],'wrong_predicted':ALPHABET[int(wrong_pred)]}
              for row,pred,blank_pred,wrong_pred in zip(valid,predictions,blank_predictions,wrong_predictions)]
        result={'name':name,'checkpoint':str(checkpoint) if checkpoint else None,
                'checkpoint_sha256':file_hash(checkpoint) if checkpoint else None,'vision_state_sha256':before,
                'train_accuracy':float((train_predictions==labels).double().mean()),
                'held_out_font_accuracy':float((predictions==targets).double().mean()),
                'case_insensitive_accuracy':sum(row['expected'].casefold()==row['predicted'].casefold() for row in rows)/len(rows),
                'blank_accuracy':float((blank_predictions==targets).double().mean()),
                'wrong_image_accuracy':float((wrong_predictions==targets).double().mean()),
                'per_character':{character:sum(row['expected']==character and row['predicted']==character for row in rows)/sum(row['expected']==character for row in rows) for character in ALPHABET},
                'results':rows}
        torch.save(head,output/(name+'-diagnostic-head.pt'));atomic_json(output/(name+'.json'),result);results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in ['results','per_character']}),flush=True)
    report={'data_manifest_sha256':file_hash(data/'manifest.json'),'config':asdict(config),
            'train_examples':len(train),'valid_examples':len(valid),'chance_accuracy':1/len(ALPHABET),
            'feature_protocol':'Frozen pre-projection pooled spatial tokens, adaptive 2x2 grid, CPU double-precision ridge; standardization/coefficients fit on training glyphs only.',
            'regularization':regularization,'seed':42,'probe_implementation_sha256':implementation,
            'encoders':[{k:v for k,v in result.items() if k not in ['results','per_character']} for result in results],
            'goal_achieved':False,'limitations':manifest['limitations']+[
                'Coarse pooled-feature separability does not prove sequence decoding or isolate every source of OCR failure.',
                'Held-out styles belong to the same public font families; real-scene/handwritten/multilingual/general visual skill is untested.']}
    if file_hash(__file__)!=implementation:raise ValueError('Probe implementation changed during this run')
    seal(report,provenance,device=device);atomic_json(output/'summary.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--data',required=True);parser.add_argument('--output');parser.add_argument('--fonts',default='/usr/share/fonts/truetype/liberation')
    parser.add_argument('--reference-manifest',default='data/rendered-text-v1/manifest.json')
    parser.add_argument('--font-notice',default='data/rendered-text-v1/font-license.txt')
    parser.add_argument('--checkpoint',action='append',default=[],help='NAME=PATH of our own encoder checkpoint')
    parser.add_argument('--device',default='cuda:3');args=parser.parse_args()
    if args.prepare:print(json.dumps(prepare(args.data,args.fonts,args.reference_manifest,args.font_notice),indent=2))
    else:
        if not args.output or not args.checkpoint:parser.error('Evaluation needs output and at least one own checkpoint')
        entries=[]
        for entry in args.checkpoint:
            name,path=entry.split('=',1)
            if not name.isidentifier() or name=='random':parser.error('Checkpoint name must be a unique safe identifier other than random')
            entries.append((name,Path(path)))
        if len({name for name,_ in entries})!=len(entries):parser.error('Duplicate checkpoint names')
        evaluate(entries,args.data,args.output,args.device)


if __name__=='__main__':main()
