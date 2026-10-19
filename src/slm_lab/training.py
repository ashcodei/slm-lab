"""Bounded pretraining, multimodal SFT, DPO, reward modeling and PPO-RLHF."""
import argparse
from contextlib import nullcontext
import copy
from dataclasses import asdict,replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image,ImageOps
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tokenizers import Tokenizer
from .data import read_jsonl,stable_id
from .corpus import PackedCorpus,file_hash
from .model import ModelConfig,TinyMultimodalLM
from .objectives import response_logps,dpo_loss,reward_loss,generalized_advantages,ppo_loss,contrastive_image_text_loss,normalize_rollout_advantages


def letterbox_box(box,dimensions,size):
    width,height=dimensions
    if len(box)!=4 or not all(math.isfinite(float(v)) for v in box):raise ValueError('Invalid normalized image box')
    resized=(size,round(height/width*size)) if width>=height else (round(width/height*size),size)
    offset=(round((size-resized[0])*.5),round((size-resized[1])*.5))
    return [round((offset[i%2]+max(0,min(1,float(value)))*resized[i%2])/size,4) for i,value in enumerate(box)]


class Encoder:
    def __init__(self, tokenizer, config, data_dir, image_size=None):
        self.tokenizer=tokenizer;self.config=config;self.data_dir=Path(data_dir)
        self.image_size=config.image_size if image_size is None else image_size
        config.image_token_count(self.image_size)
        self.bos=tokenizer.token_to_id('<bos>');self.eos=tokenizer.token_to_id('<eos>')
        self.pad=tokenizer.token_to_id('<pad>')
        if any(x is None for x in [self.bos,self.eos,self.pad]):
            raise ValueError('Tokenizer needs <bos>, <eos> and <pad> special tokens.')
        if tokenizer.get_vocab_size()!=config.vocab_size:
            raise ValueError('Model vocabulary must equal the newly trained tokenizer vocabulary.')
        self.truncated=0

    def tokens(self,text):
        return self.tokenizer.encode(text).ids

    def prompt(self,row):
        text=row['prompt']
        if row.get('image') and row.get('task') in {'region_ocr','document_region_ocr','object_region'}:
            with Image.open(self.data_dir/row['image']) as image:
                box=letterbox_box(row['normalized_box'],image.size,self.image_size)
            if row['task']=='object_region':
                text='Identify the object inside the box '+str(box)+'. Coordinates are normalized [left, top, right, bottom] in the padded input image. Answer with the object class name only.'
            else:
                text='Read the text inside the box '+str(box)+'. Coordinates are normalized [left, top, right, bottom] in the padded input image.'
        return text if text.rstrip().endswith('Assistant:') else 'User: '+text+'\nAssistant:'

    def sequence(self,row,response_field=None):
        limit=self.config.max_text_length
        if 'tokens' in row:
            ids=row['tokens']
            if response_field is not None or not 2<=len(ids)<=limit:
                raise ValueError('Packed token sequences are only supported for pretraining.')
            return ids,ids.copy()
        if response_field is None:
            body=self.tokens(row['text'])
            self.truncated+=int(len(body)+2>limit)
            ids=[self.bos]+body[:limit-2]+[self.eos]
            return ids,ids.copy()
        prompt=self.tokens(self.prompt(row))
        response=self.tokens(row[response_field])
        # A bounded research run records truncation explicitly. At production
        # scale pre-pack documents and select a length appropriate to the task.
        max_response=max(1,limit//2-1)
        self.truncated+=int(len(response)>max_response or len(prompt)+len(response)+2>limit)
        response=response[:max_response]+[self.eos]
        reserved=len(response)
        if 'chosen' in row and 'rejected' in row:
            # Both preference responses must be judged under identical retained
            # prompt tokens, including when their response lengths differ.
            reserved=max(min(len(self.tokens(row[key])),max_response)+1 for key in ['chosen','rejected'])
        prompt=prompt[-(limit-reserved-1):]
        ids=[self.bos]+prompt+response
        labels=[-100]*(len(prompt)+1)+response
        return ids,labels

    def pixels(self,row):
        if not row.get('image'):return None
        with Image.open(self.data_dir/row['image']) as image:
            image=ImageOps.pad(image.convert('RGB'),(self.image_size,)*2,
                               method=Image.Resampling.BICUBIC,color=(128,128,128))
            arr=np.asarray(image,dtype=np.float32)/127.5-1
        return torch.from_numpy(arr.copy()).permute(2,0,1)

    def batch(self,rows,field,device):
        sequences=[self.sequence(row,field) for row in rows]
        length=max(len(x[0]) for x in sequences)
        ids=torch.full((len(rows),length),self.pad,dtype=torch.long)
        labels=torch.full_like(ids,-100);mask=torch.zeros_like(ids,dtype=torch.bool)
        for i,(seq,target) in enumerate(sequences):
            ids[i,:len(seq)]=torch.tensor(seq);labels[i,:len(seq)]=torch.tensor(target)
            mask[i,:len(seq)]=True
        images=[self.pixels(row) for row in rows]
        if any(x is not None for x in images) and not all(x is not None for x in images):
            raise ValueError('A minibatch must contain either all image rows or all text rows.')
        pixels=torch.stack(images).to(device) if images[0] is not None else None
        return {'input_ids':ids.to(device),'labels':labels.to(device),
                'attention_mask':mask.to(device),'pixel_values':pixels}


def load_checkpoint(path,device='cpu',training_state=True):
    # Inference from a resumable snapshot should not allocate its Adam/reference
    # tensors on a GPU. Memory mapping touches only the model pages we need.
    obj=torch.load(path,map_location=device if training_state else 'cpu',weights_only=True,
                   mmap=not training_state)
    model=TinyMultimodalLM(ModelConfig(**obj['config']))
    if training_state:model=model.to(device)
    model.load_state_dict(obj.pop('model'))
    if not training_state:
        for key in ['optimizer','reference_model','torch_rng_state','cuda_rng_state']:obj.pop(key,None)
        model=model.to(device)
    return model,obj


def last_scores(out,batch):
    last=batch['attention_mask'].sum(-1)-1
    return out['values'].gather(1,last[:,None]).squeeze(1)


def forward(model,batch,return_logits=True):
    return model(**{k:v for k,v in batch.items() if k!='labels'},return_logits=return_logits)


def freeze(model):
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


def make_rollout(policy,reference,reward_model,encoder,row,device,max_new_tokens,kl_coef):
    body=encoder.tokens(encoder.prompt(row))[-(policy.config.max_text_length-max_new_tokens-1):]
    prompt=torch.tensor([[encoder.bos]+body],device=device)
    pixels=encoder.pixels(row)
    if pixels is not None:pixels=pixels[None].to(device)
    with torch.no_grad():
        # Temperature is exactly 1: stored behavior probabilities must equal the
        # actual rollout distribution for the PPO importance ratio to be valid.
        rollout=policy.generate(prompt,pixels,max_new_tokens=max_new_tokens,temperature=1.0,
                                eos_token_id=encoder.eos,return_rollout=True)
        ids=rollout['tokens'];start=prompt.shape[1]-1
        old_logps=rollout['logps'][0];values=rollout['values'][0]
        ref_logits=reference(ids,pixels)['logits'][0,start:-1].float()
        ref_logps=ref_logits.log_softmax(-1).gather(1,ids[0,start+1:,None]).squeeze(1)
        terminal=reward_model(ids,pixels,return_logits=False)['values'][0,-1].float()
        rewards=-kl_coef*(old_logps-ref_logps)
        rewards[-1]+=terminal
        advantages,returns=generalized_advantages(rewards,values)
    return {'ids':ids,'pixels':pixels,'start':start,'old_logps':old_logps,'old_values':values,
            'advantages':advantages,'returns':returns,'reward':float(terminal)}


class TrainingOutputs(torch.nn.Module):
    """Expose only tensors consumed by the stage to DDP's unused-parameter search."""
    def __init__(self,model,stage):
        super().__init__();self.model=model;self.stage=stage

    def forward(self,*args,**kwargs):
        if self.stage=='contrastive':
            if args:raise ValueError('Contrastive encoder expects keyword inputs.')
            with torch.no_grad():
                text=self.model(input_ids=kwargs['input_ids'],attention_mask=kwargs['attention_mask'],
                                return_hidden=True,return_logits=False)['hidden']
                last=kwargs['attention_mask'].sum(-1)-1
                text=text[torch.arange(text.shape[0],device=text.device),last]
            images=self.model.vision(kwargs['pixel_values'],self.model.gradient_checkpointing).mean(1)
            return {'image_features':images,'text_features':text}
        if self.stage=='reward':kwargs['return_logits']=False
        out=self.model(*args,**kwargs)
        keys={'pretrain':['loss'],'sft':['loss'],'dpo':['logits'],
              'reward':['values'],'ppo':['logits','values']}[self.stage]
        return {k:out[k] for k in keys}


def supervised_gradients(model,encoder,groups,field,device,amp,world=1):
    """Accumulate the global token-mean gradient, including unequal response lengths."""
    batches=[encoder.batch(group,field,device) for group in groups]
    counts=[(batch['labels'][:,1:]!=-100).sum().float() for batch in batches]
    total=torch.stack(counts).sum()
    if world>1:dist.all_reduce(total)
    if total<=0:raise ValueError('No supervised tokens in this optimizer step.')
    loss_sum=torch.zeros((),device=device,dtype=torch.float32)
    for micro,(batch,count) in enumerate(zip(batches,counts)):
        sync=model.no_sync() if world>1 and micro<len(batches)-1 else nullcontext()
        with sync:
            with amp():loss=model(**batch)['loss']
            if not torch.isfinite(loss):raise ValueError('Nonfinite supervised loss; stopping before update.')
            # DDP averages workers' gradients. Correct for that average so
            # variable-length responses have equal weight per target token.
            (loss*(world*count/total)).backward()
        loss_sum+=loss.detach()*count
    if world>1:dist.all_reduce(loss_sum)
    return loss_sum/total,int(total)


def train(args):
    args.gradient_accumulation=getattr(args,'gradient_accumulation',1)
    args.trainable=getattr(args,'trainable','all')
    args.contrastive_temperature=getattr(args,'contrastive_temperature',.07)
    args.image_size=getattr(args,'image_size',None)
    if args.contrastive_temperature<=0:raise ValueError('Contrastive temperature must be positive.')
    if args.gradient_accumulation<1:raise ValueError('Gradient accumulation must be positive.')
    if args.gradient_accumulation!=1 and args.stage not in {'pretrain','sft'}:
        raise ValueError('Gradient accumulation currently supports pretraining and SFT.')
    if args.steps<1 or args.batch_size<1 or args.learning_rate<=0:
        raise ValueError('Steps, batch size and learning rate must be positive.')
    if args.checkpoint_every<0 or args.keep_checkpoints<1:
        raise ValueError('Checkpoint interval must be nonnegative and retention positive.')
    if args.resume and not args.checkpoint:
        raise ValueError('--resume requires a training-state --checkpoint.')
    if args.threads<1 or args.log_every<1 or args.ppo_epochs<1:
        raise ValueError('Thread count, logging interval and PPO epochs must be positive.')
    if args.kl_coef<0 or args.beta<=0 or args.max_new_tokens<1:
        raise ValueError('KL coefficient must be nonnegative; beta and rollout length positive.')
    torch.set_num_threads(args.threads)
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    local_rank=int(os.environ.get('LOCAL_RANK','0'))
    device=torch.device(args.device if world==1 or args.device=='cpu' else f'cuda:{local_rank}')
    if device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('No usable CUDA GPU is exposed. Request a GPU-enabled instance first.')
    if world>1:
        if args.stage=='ppo':raise ValueError('PPO rollout implementation currently supports one GPU.')
        if device.type=='cuda':torch.cuda.set_device(device)
        dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    random.seed(args.seed);torch.manual_seed(args.seed)
    output=Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory; checkpoints are not overwritten.')
    output.mkdir(parents=True,exist_ok=True)
    tokenizer=Tokenizer.from_file(args.tokenizer)
    token_hash=hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
    if args.checkpoint:
        raw,prior=load_checkpoint(args.checkpoint,device)
        if prior['tokenizer_sha256']!=token_hash:
            raise ValueError('Checkpoint tokenizer does not match the supplied tokenizer.')
    else:
        if args.stage not in {'pretrain','sft'}:
            raise ValueError('Alignment stages require an existing trained --checkpoint.')
        config=replace(ModelConfig.load(args.config),vocab_size=tokenizer.get_vocab_size())
        raw=TinyMultimodalLM(config).to(device)
    if args.stage=='contrastive' and args.trainable!='vision':
        raise ValueError('Contrastive warmup must freeze the language component with --trainable vision.')
    if args.trainable=='vision' and args.stage not in {'sft','contrastive'}:
        raise ValueError('Vision-only warmup requires image-caption SFT data.')
    if args.trainable!='all':
        for name,parameter in raw.named_parameters():
            parameter.requires_grad_(name.startswith('vision.') if args.trainable=='vision' else not name.startswith('vision.'))
    raw.gradient_checkpointing=args.gradient_checkpointing
    raw.train()
    if args.image_size is None:
        args.image_size=prior.get('training_options',{}).get('image_size',raw.config.image_size) if args.checkpoint else raw.config.image_size
    encoder=Encoder(tokenizer,raw.config,Path(args.data).parent,args.image_size)
    data_path=Path(args.data)
    if data_path.is_dir():
        if args.stage!='pretrain':raise ValueError('Packed corpora are only for pretraining.')
        rows=PackedCorpus(data_path,token_hash,raw.config.max_text_length)
        data_hash=file_hash(data_path/'corpus.json')
    else:
        rows=read_jsonl(args.data)
        data_hash=file_hash(data_path)
    if args.trainable=='vision' and any(not row.get('image') for row in rows):
        raise ValueError('Vision-only warmup requires an image in every record.')
    if args.stage in {'dpo','reward'}:
        if any(r.get('feedback_source') not in {'human','ai','synthetic'} for r in rows):
            raise ValueError('Preference records must identify human, AI or synthetic feedback.')
    if args.stage=='ppo' and args.max_new_tokens>=raw.config.max_text_length-1:
        raise ValueError('PPO requires room for both prompt and completion.')
    reference=freeze(copy.deepcopy(raw)) if args.stage in {'dpo','ppo'} else None
    reward_model=None
    if args.stage=='ppo':
        if not args.reward_checkpoint:raise ValueError('PPO requires a learned --reward-checkpoint.')
        reward_model,reward_meta=load_checkpoint(args.reward_checkpoint,device)
        if reward_meta.get('stage')!='reward' or reward_meta['tokenizer_sha256']!=token_hash:
            raise ValueError('Reward checkpoint must be a reward model with the same tokenizer.')
        if reward_model.config!=raw.config:raise ValueError('Reward and policy architecture must match.')
        if 'human' not in reward_meta.get('feedback_sources',[]):
            raise ValueError('RLHF requires a reward model trained on human preferences.')
        freeze(reward_model)
    model=DistributedDataParallel(TrainingOutputs(raw,args.stage),device_ids=[local_rank] if device.type=='cuda' else None,
                                 find_unused_parameters=True) if world>1 else (TrainingOutputs(raw,args.stage) if args.stage=='contrastive' else raw)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.learning_rate,weight_decay=0.1)
    start_step=0
    if args.resume:
        if args.stage in {'reward','ppo'} and device.type=='cuda' and prior.get('critic_projection_precision')!='fp32':
            raise ValueError('Resume requires the same FP32 critic projection; inspect earlier mixed-precision states.')
        if prior.get('stage')!=args.stage or 'optimizer' not in prior:
            raise ValueError('Resume requires a training-state checkpoint from the same stage.')
        if prior['data_sha256']!=data_hash or prior['planned_steps']!=args.steps:
            raise ValueError('Resume requires the same data and total step budget.')
        if prior['training_options'].get('gradient_accumulation',1)!=args.gradient_accumulation:
            raise ValueError('Resume requires the same gradient accumulation.')
        if prior['training_options'].get('image_size',raw.config.image_size)!=args.image_size:
            raise ValueError('Resume requires the same image resolution.')
        if (not isinstance(rows,PackedCorpus) and any(r.get('task') in {'region_ocr','document_region_ocr','object_region'} for r in rows)
            and prior.get('image_prompt_coordinates')!='letterbox input'):
            raise ValueError('Resume requires the same image-box coordinate transform.')
        if prior['training_options']!={k:getattr(args,k) for k in prior['training_options']}:
            raise ValueError('Resume requires the same training hyperparameters.')
        if prior.get('world_size',1)!=world:raise ValueError('Resume requires the same worker count.')
        if prior.get('trainable','all')!=args.trainable:raise ValueError('Resume requires the same trainable components.')
        if reference is not None:
            reference.load_state_dict(prior['reference_model'])
        if args.stage=='ppo' and prior.get('rollout_probability_source')!='generation logits':
            raise ValueError('PPO resume requires checkpoints that stored actual sampling probabilities.')
        if args.stage=='ppo' and prior.get('ppo_advantage_normalization')!='rollout batch':
            raise ValueError('PPO resume requires the same rollout-batch advantage normalization.')
        if args.stage=='ppo' and prior.get('reward_checkpoint_sha256')!=file_hash(args.reward_checkpoint):
            raise ValueError('Resume requires the same reward model.')
        opt.load_state_dict(prior['optimizer']);start_step=prior['steps']
        encoder.truncated=prior.get('truncated_sequences',0)
        if start_step>=args.steps:raise ValueError('This checkpoint already completed its step budget.')
    use_bf16=device.type=='cuda' and torch.cuda.is_bf16_supported()
    def amp():return torch.autocast(device_type='cuda',dtype=torch.bfloat16) if use_bf16 else nullcontext()
    if isinstance(rows,PackedCorpus):
        # An affine permutation visits every packed window once per epoch,
        # without allocating a billion-element shuffled index array.
        size=len(rows);offset=random.randrange(size);stride=random.randrange(1,size+1)
        while math.gcd(stride,size)!=1:stride=stride%size+1
        def index(position):return (offset+position*stride)%size
    else:
        order=list(range(len(rows)));random.shuffle(order)
        image_count=sum(bool(row.get('image')) for row in rows)
        if 0<image_count<len(rows):
            if args.stage!='sft':raise ValueError('Mixed text/image records are supported for SFT only.')
            # Shuffle complete global minibatches, preserving each batch's
            # image/text modality. All workers use the same modality per microstep.
            width=args.batch_size*world
            buckets=[[i for i in order if bool(rows[i].get('image'))==flag] for flag in [False,True]]
            minibatches=[bucket[start:start+width] for bucket in buckets
                         for start in range(0,len(bucket)-width+1,width)]
            if not minibatches:raise ValueError('Mixed SFT needs a full global batch in each modality.')
            if any(len(bucket)<width for bucket in buckets):raise ValueError('Each modality needs a full global batch.')
            random.shuffle(minibatches);order=[i for batch in minibatches for i in batch]
        def index(position):return order[position%len(order)]
    metrics=prior.get('metrics',[]).copy() if args.resume else []
    if args.resume:
        torch.set_rng_state(prior['torch_rng_state'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state(prior['cuda_rng_state'].cpu(),device)
    started=time.monotonic()
    feedback=[] if isinstance(rows,PackedCorpus) else sorted({r.get('feedback_source') for r in rows if r.get('feedback_source')})
    if rank==0:tokenizer.save(str(output/'tokenizer.json'))
    if args.stage=='ppo':feedback=reward_meta['feedback_sources']
    def save(path,completed,include_optimizer=True):
        state={k:v.detach().cpu() for k,v in raw.state_dict().items()}
        obj={'model':state,'config':asdict(raw.config),'stage':args.stage,'seed':args.seed,
            'critic_projection_precision':'fp32',
            'tokenizer_sha256':token_hash,'data_sha256':data_hash,'feedback_sources':feedback,
            'image_prompt_coordinates':'letterbox input',
            'steps':completed,'planned_steps':args.steps,'world_size':world,'trainable':args.trainable,'truncated_sequences':encoder.truncated,
            'metrics':metrics,'torch_rng_state':torch.get_rng_state(),
            'training_options':{k:getattr(args,k) for k in ['batch_size','learning_rate','seed',
                'beta','kl_coef','ppo_epochs','max_new_tokens','gradient_checkpointing','gradient_accumulation','contrastive_temperature','image_size']}}
        if include_optimizer:
            obj['optimizer']=opt.state_dict()
            if reference is not None:
                obj['reference_model']={k:v.detach().cpu() for k,v in reference.state_dict().items()}
        if args.stage=='ppo':
            obj['reward_checkpoint_sha256']=file_hash(args.reward_checkpoint)
            obj['rollout_probability_source']='generation logits'
            obj['ppo_advantage_normalization']='rollout batch'
        if device.type=='cuda':obj['cuda_rng_state']=torch.cuda.get_rng_state(device)
        temporary=path.with_suffix('.tmp')
        torch.save(obj,temporary);os.replace(temporary,path)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    for step in range(start_step,args.steps):
        if device.type=='cuda':torch.cuda.synchronize(device)
        step_started=time.monotonic()
        groups=[[rows[index((step*args.gradient_accumulation+micro)*args.batch_size*world+rank*args.batch_size+i)]
                 for i in range(args.batch_size)] for micro in range(args.gradient_accumulation)]
        group=groups[0]
        opt.zero_grad(set_to_none=True)
        lr=args.learning_rate*min(1,(step+1)/max(1,args.steps//10))
        # Warmup then cosine decay; no implicit infinite training loop.
        lr*=0.1+0.9*0.5*(1+math.cos(math.pi*step/args.steps))
        for param_group in opt.param_groups:param_group['lr']=lr
        with amp():
            if args.stage in {'pretrain','sft'}:
                loss,target_tokens=supervised_gradients(model,encoder,groups,
                    None if args.stage=='pretrain' else 'response',device,amp,world)
            elif args.stage=='contrastive':
                batch=encoder.batch([dict(row,text=row['response']) for row in group],None,device)
                output_features=model(**batch)
                images=output_features['image_features'];texts=output_features['text_features']
                caption_ids=torch.tensor([int(stable_id(row['response'])[:15],16) for row in group],device=device)
                if world>1:
                    from torch.distributed.nn.functional import all_gather as differentiable_gather
                    images=torch.cat(differentiable_gather(images));text_parts=[torch.empty_like(texts) for _ in range(world)]
                    id_parts=[torch.empty_like(caption_ids) for _ in range(world)]
                    dist.all_gather(text_parts,texts);dist.all_gather(id_parts,caption_ids)
                    texts=torch.cat(text_parts);caption_ids=torch.cat(id_parts)
                if caption_ids.unique().numel()<2:raise ValueError('Contrastive batch needs at least two distinct captions.')
                loss=contrastive_image_text_loss(images,texts,caption_ids,args.contrastive_temperature)
            elif args.stage in {'dpo','reward'}:
                pair_rows=[dict(r,response=r['chosen']) for r in group]+[dict(r,response=r['rejected']) for r in group]
                joined=encoder.batch(pair_rows,'response',device)
                count=len(group)
                good={k:(v[:count] if v is not None else None) for k,v in joined.items()}
                bad={k:(v[count:] if v is not None else None) for k,v in joined.items()}
                outputs=forward(model,joined)
                chosen={k:v[:count] for k,v in outputs.items()}
                rejected={k:v[count:] for k,v in outputs.items()}
                if args.stage=='reward':
                    loss=reward_loss(last_scores(chosen,good),last_scores(rejected,bad))
                else:
                    with torch.no_grad():
                        rc=forward(reference,good);rr=forward(reference,bad)
                    loss=dpo_loss(response_logps(chosen['logits'],good['input_ids'],good['labels']),
                                  response_logps(rejected['logits'],bad['input_ids'],bad['labels']),
                                  response_logps(rc['logits'],good['input_ids'],good['labels']),
                                  response_logps(rr['logits'],bad['input_ids'],bad['labels']),args.beta)
            else:
                rollout=[make_rollout(raw,reference,reward_model,encoder,row,device,
                                      args.max_new_tokens,args.kl_coef) for row in group]
                normalize_rollout_advantages(rollout)
                # Multiple epochs use fixed behavior log-probabilities and rewards.
                for epoch in range(args.ppo_epochs):
                    opt.zero_grad(set_to_none=True)
                    losses=[]
                    for r in rollout:
                        out=model(r['ids'],r['pixels']);start=r['start']
                        logps=out['logits'][0,start:-1].float().log_softmax(-1).gather(
                            1,r['ids'][0,start+1:,None]).squeeze(1)
                        losses.append(ppo_loss(logps,r['old_logps'],r['advantages'],
                            out['values'][0,start:-1].float(),r['old_values'],r['returns']))
                    loss=torch.stack(losses).mean()
                    if not torch.isfinite(loss):raise ValueError('Nonfinite PPO loss; stopping before update.')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);opt.step()
        if args.stage!='ppo':
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss; stopping before update.')
            if args.stage not in {'pretrain','sft'}:loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);opt.step()
        if device.type=='cuda':torch.cuda.synchronize(device)
        elapsed=time.monotonic()-step_started
        if world>1:
            duration=torch.tensor(elapsed,device=device,dtype=torch.float64)
            dist.all_reduce(duration,op=dist.ReduceOp.MAX);elapsed=float(duration)
        metric={'step':step+1,'loss':float(loss.detach()),'learning_rate':lr,'step_seconds':elapsed}
        if args.stage in {'pretrain','sft'}:
            metric['global_target_tokens']=target_tokens
            metric['global_target_tokens_per_second']=target_tokens/elapsed
        if args.stage=='contrastive':
            metric['global_images']=args.batch_size*world
            metric['global_images_per_second']=args.batch_size*world/elapsed
        if args.stage=='ppo':metric['reward']=sum(r['reward'] for r in rollout)/len(rollout)
        metrics.append(metric)
        if rank==0 and args.checkpoint_every and (step+1)%args.checkpoint_every==0:
            save(output/f'step-{step+1:08d}.pt',step+1)
            old=sorted(output.glob('step-*.pt'))[:-args.keep_checkpoints]
            for stale in old:stale.unlink()
        if rank==0 and ((step+1)%args.log_every==0 or step==0):
            print(json.dumps(metric),flush=True)
            progress={'status':'training','step':step+1,'planned_steps':args.steps,
                'elapsed_seconds':time.monotonic()-started,'latest_metric':metric,
                'processed_target_tokens':sum(m.get('global_target_tokens',0) for m in metrics),
                'data_sha256':data_hash,'tokenizer_sha256':token_hash,'world_size':world}
            temporary=output/'progress.tmp'
            temporary.write_text(json.dumps(progress,indent=2)+'\n')
            os.replace(temporary,output/'progress.json')
    if rank==0:
        save(output/'model.pt',args.steps,include_optimizer=False)
        save(output/'training-state.pt',args.steps)
        tokenizer.save(str(output/'tokenizer.json'))
        report={'stage':args.stage,'parameters':raw.parameter_counts(),'device':str(device),
            'critic_projection_precision':'fp32',
            'data_sha256':data_hash,'tokenizer_sha256':token_hash,'planned_steps':args.steps,
            'image_size':args.image_size,'visual_tokens':raw.config.image_token_count(args.image_size),
            'image_prompt_coordinates':'letterbox input',
            'world_size':world,'trainable':args.trainable,'trainable_parameters':sum(p.numel() for p in raw.parameters() if p.requires_grad),'gradient_accumulation':args.gradient_accumulation,'bf16':use_bf16,'steps':args.steps,'elapsed_seconds':time.monotonic()-started,
            'feedback_sources':feedback,'truncated_sequences':encoder.truncated,'metrics':metrics,
            'gpu_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
            'status':'training run completed; general capability unverified'}
        (output/'training.json').write_text(json.dumps(report,indent=2)+'\n')
        progress={'status':'training complete; capability unverified','step':args.steps,
            'planned_steps':args.steps,'elapsed_seconds':report['elapsed_seconds'],'latest_metric':metrics[-1],
            'processed_target_tokens':sum(m.get('global_target_tokens',0) for m in metrics),
            'data_sha256':data_hash,'tokenizer_sha256':token_hash,'world_size':world}
        (output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    if world>1:dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['pretrain','sft','contrastive','dpo','reward','ppo'])
    p.add_argument('--config',default='configs/model.json')
    p.add_argument('--tokenizer',required=True);p.add_argument('--data',required=True)
    p.add_argument('--output',required=True);p.add_argument('--checkpoint')
    p.add_argument('--reward-checkpoint');p.add_argument('--device',default='cpu')
    p.add_argument('--steps',type=int,default=100);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--gradient-accumulation',type=int,default=1)
    p.add_argument('--trainable',choices=['all','vision','language'],default='all')
    p.add_argument('--contrastive-temperature',type=float,default=.07)
    p.add_argument('--image-size',type=int,help='Square image resolution; defaults to the previous checkpoint setting.')
    p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--beta',type=float,default=0.1);p.add_argument('--kl-coef',type=float,default=0.05)
    p.add_argument('--ppo-epochs',type=int,default=2);p.add_argument('--max-new-tokens',type=int,default=16)
    p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--gradient-checkpointing',action='store_true')
    p.add_argument('--checkpoint-every',type=int,default=1000)
    p.add_argument('--keep-checkpoints',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    try:train(args)
    except ValueError as e:p.error(str(e))


if __name__=='__main__':main()
