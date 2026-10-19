"""CPU text/image queries of our own independently stored dense checkpoints.

This is an inference interface, not a model-quality gate. It never loads a
teacher, ancestor checkpoint, optimizer, external model, or repair service.
Artifact inspection validates envelopes/checksums, not every neural tensor.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import torch
from tokenizers import Tokenizer

from .corpus import file_hash
from .inference_rotary_cache_v1 import inference_cache
from .own_dense_standalone_v1 import FORMAT as DENSE_FORMAT,load as load_dense
from .own_actor_dense_fold_v1 import FORMAT as FOLDED_FORMAT,load as load_folded


def inspect_package(checkpoint,*,expected_sha256=None,expected_tokenizer_sha256=None,expected_sources=None):
    path=Path(checkpoint).expanduser().resolve();digest=file_hash(path)
    if expected_sha256 is not None and expected_sha256!=digest:raise ValueError('Checkpoint checksum differs from the requested identity')
    raw=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if raw.get('format') not in {DENSE_FORMAT,FOLDED_FORMAT}:raise ValueError('A complete own standalone dense model is required; actor/optimizer/legacy checkpoints are unsupported')
    if raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False:
        raise ValueError('The supported research artifact must retain its original explicit completion flags')
    payload=raw['complete_dense'];sources=payload['sources'];config=payload['config'];counts=payload['actual_counts']
    if (not isinstance(sources,dict) or not sources or any(not isinstance(k,str) or not isinstance(v,str)
            or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for k,v in sources.items())):
        raise ValueError('Explicit own model source identities required')
    if expected_sources is not None and sources!=expected_sources:raise ValueError('Checkpoint source identities differ from the supplied source map')
    token_path=path.parent/'tokenizer.json';token_digest=file_hash(token_path)
    if expected_tokenizer_sha256 is not None and token_digest!=expected_tokenizer_sha256:
        raise ValueError('Packaged tokenizer checksum differs from the requested identity')
    tokenizer=Tokenizer.from_file(str(token_path))
    if (tokenizer.get_vocab_size()!=config['vocab_size'] or [tokenizer.token_to_id(s) for s in ['<pad>','<bos>','<eos>']]!=[0,1,2]):
        raise ValueError('One complete packaged native vocabulary and original special token IDs required')
    if (type(counts.get('total_unique')) is not int or not 0<counts['total_unique']<10**9
            or type(raw.get('completed_updates')) is not int or raw['completed_updates']<0):
        raise ValueError('An explicitly counted own under1B research checkpoint and actual training cursor required')
    if raw['format']==DENSE_FORMAT:
        if raw.get('tokenizer_sha256')!=token_digest or raw.get('old_checkpoint_optimizer_or_training_reference_needed_for_inference') is not False:
            raise ValueError('Complete own model must package its exact tokenizer and require no ancestor checkpoint')
    elif (raw.get('old_base_or_actor_checkpoint_or_teacher_needed_for_inference') is not False
            or raw.get('optimizer_or_training_reference_included') is not False):
        raise ValueError('Folded own model must be independent of an old actor/base/optimizer')
    if file_hash(path)!=digest:raise ValueError('Checkpoint bytes changed during inspection')
    return dict(checkpoint=str(path),checkpoint_sha256=digest,format=raw['format'],tokenizer_sha256=token_digest,
        source_identities=sources,source_map_verified_against_external_input=expected_sources is not None,
        checksum_verified_against_external_input=expected_sha256 is not None,
        tokenizer_checksum_verified_against_external_input=expected_tokenizer_sha256 is not None,
        declared_counts=counts,declared_config=config,completed_updates=raw['completed_updates'],
        inspect_only=True,registered_parameter_count_and_tensor_values_not_checked_by_inspection=True,
        model_quality_not_proven_by_loading=True)


def load_package(package):
    loader=load_dense if package['format']==DENSE_FORMAT else load_folded
    model,metadata=loader(package['checkpoint'],expected_checkpoint_sha256=package['checkpoint_sha256'],expected_sources=package['source_identities'])
    actual=sum(p.numel() for p in model.parameters())
    if (actual!=package['declared_counts']['total_unique'] or actual>=10**9
            or model.counts()!=package['declared_counts'] or torch.cuda.is_initialized()):
        raise ValueError('Every actual own registered parameter must agree with the complete CPU under1B package')
    if file_hash(Path(package['checkpoint']).parent/'tokenizer.json')!=package['tokenizer_sha256']:
        raise ValueError('Packaged native tokenizer changed after inspection')
    return model.eval(),dict(metadata,actual_registered_parameters=actual)


def prompt_ids(tokenizer,config,prompt,*,raw_prompt=False,max_new_tokens=128):
    if (not isinstance(prompt,str) or not prompt.strip() or type(max_new_tokens) is not int
            or not 1<=max_new_tokens<=config.max_text_length):raise ValueError('Nonempty complete text and a positive bounded native action budget required')
    text=prompt if raw_prompt else 'User: '+prompt+'\nAssistant:'
    # Insert own BOS explicitly; no tokenizer postprocessor adds boundaries.
    ids=[1,*tokenizer.encode(text,add_special_tokens=False).ids]
    if len(ids)+max_new_tokens>config.max_text_length:
        raise ValueError('Complete prompt plus response budget exceeds context; shorten the prompt or action budget')
    return torch.tensor([ids],dtype=torch.long),text


def image_pixels(path,config,size):
    if path is None:return None,None
    from PIL import Image,ImageOps
    import numpy as np
    if size is None:size=config.max_image_size
    config.image_token_count(size)
    path=Path(path).expanduser().resolve();digest=file_hash(path)
    with Image.open(path) as original:
        image=ImageOps.pad(original.convert('RGB'),(size,size),method=Image.Resampling.BICUBIC,color=(128,128,128))
        values=np.asarray(image,dtype=np.float32)/127.5-1
    if file_hash(path)!=digest:raise ValueError('Image bytes changed during preprocessing')
    pixels=torch.from_numpy(values.copy()).permute(2,0,1)[None]
    return pixels,dict(path=str(path),source_file_sha256=digest,image_size=size,
        pixel_tensor_sha256=__import__('hashlib').sha256(pixels.contiguous().numpy().tobytes()).hexdigest(),
        visual_prefix_tokens=config.image_token_count(size))


def query(model,tokenizer,prompt,*,max_new_tokens=128,raw_prompt=False,image=None,image_size=None,precision='bf16'):
    if precision not in {'bf16','fp32'}:raise ValueError('Explicit CPU bf16 or fp32 precision required')
    if (next(model.parameters()).device.type!='cpu' or any(p.device.type!='cpu' for p in model.parameters())):
        raise ValueError('This interface uses a complete own CPU model')
    ids,text=prompt_ids(tokenizer,model.config,prompt,raw_prompt=raw_prompt,max_new_tokens=max_new_tokens)
    pixels,image_metadata=image_pixels(image,model.config,image_size);model.eval();start=time.perf_counter()
    amp=torch.autocast('cpu',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()
    rng=torch.get_rng_state().clone()
    with torch.no_grad(),amp,inference_cache():
        tokens=model.generate(ids,pixels,max_new_tokens=max_new_tokens,eos_token_id=2,use_cache=True)
    if (tokens.ndim!=2 or tokens.shape[0]!=1 or not torch.equal(tokens[:,:ids.shape[1]],ids)
            or not 1<=tokens.shape[1]-ids.shape[1]<=max_new_tokens or not torch.equal(rng,torch.get_rng_state())):
        raise ValueError('Cached greedy generation must preserve its complete prefix, bounded actions and caller RNG')
    actions=tokens[0,ids.shape[1]:].tolist()
    return dict(response=tokenizer.decode(actions),actual_generated_token_ids=actions,complete_prompt_token_ids=ids[0].tolist(),
        complete_prompt=text,prompt_mode='raw_completion' if raw_prompt else 'native_User_Assistant',
        requested_native_action_budget=max_new_tokens,actual_native_actions=len(actions),actual_stopped_on_EOS=actions[-1]==2,
        image=image_metadata,inference_device='cpu',inference_precision=precision,generation_seconds=time.perf_counter()-start,
        generated_text_returned_without_repair=True,external_model_or_teacher_used=False,model_quality_not_proven_by_loading=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Query our own self-contained under1B dense model on CPU. Responses are native model output.')
    parser.add_argument('--checkpoint',required=True,help='Complete standalone model.pt, packaged beside tokenizer.json')
    parser.add_argument('--inspect',action='store_true',help='Inspect identity and declared metadata without constructing the full neural model')
    parser.add_argument('--prompt');parser.add_argument('--raw-prompt',action='store_true',help='Use complete prompt verbatim for code/text completion')
    parser.add_argument('--image',help='Optional local image, using the original aspect-preserving gray-pad preprocessing')
    parser.add_argument('--image-size',type=int);parser.add_argument('--max-new-tokens',type=int,default=128)
    parser.add_argument('--precision',choices=['bf16','fp32'],default='bf16');parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--expected-sha256');parser.add_argument('--expected-tokenizer-sha256')
    parser.add_argument('--expected-sources',help='Optional JSON source-name to SHA256 map to verify against an external receipt')
    parser.add_argument('--json',action='store_true',help='Emit raw token IDs, stopping boundary, source identities and runtime metadata')
    args=parser.parse_args(argv)
    if args.threads<1:parser.error('--threads must be positive')
    if not args.inspect and args.prompt is None:parser.error('--prompt is required unless --inspect is selected')
    if args.image_size is not None and args.image is None:parser.error('--image-size requires --image')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    expected=None if args.expected_sources is None else json.loads(Path(args.expected_sources).read_text())
    package=inspect_package(args.checkpoint,expected_sha256=args.expected_sha256,
        expected_tokenizer_sha256=args.expected_tokenizer_sha256,expected_sources=expected)
    if args.inspect:print(json.dumps(package,indent=2));return package
    model,metadata=load_package(package);tokenizer=Tokenizer.from_file(str(Path(package['checkpoint']).parent/'tokenizer.json'))
    result=query(model,tokenizer,args.prompt,max_new_tokens=args.max_new_tokens,raw_prompt=args.raw_prompt,
        image=args.image,image_size=args.image_size,precision=args.precision)
    result.update(checkpoint_sha256=package['checkpoint_sha256'],tokenizer_sha256=package['tokenizer_sha256'],
        actual_registered_parameters=metadata['actual_registered_parameters'],source_map_verified_against_external_input=package['source_map_verified_against_external_input'],
        completed_checkpoint_updates=package['completed_updates'])
    print(json.dumps(result,indent=2) if args.json else result['response']);return result


if __name__=='__main__':main()
