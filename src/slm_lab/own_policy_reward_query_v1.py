"""Query the own standalone unfused policy and its shared trained reward."""
import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .own_native_query_v1 import query,prompt_ids
from .own_policy_reward_standalone_v1 import inspect_package,load


def main(argv=None):
    parser=argparse.ArgumentParser(description='Query the own standalone text/image policy and preserved shared reward.')
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--inspect',action='store_true')
    parser.add_argument('--prompt');parser.add_argument('--raw-prompt',action='store_true')
    parser.add_argument('--image');parser.add_argument('--image-size',type=int)
    parser.add_argument('--max-new-tokens',type=int,default=128);parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--score-response',help='Score a complete supplied text response with the original FP32human-ranking head')
    parser.add_argument('--precision',choices=['fp32','bf16'],default='fp32');parser.add_argument('--json',action='store_true')
    parser.add_argument('--expected-sha256');parser.add_argument('--expected-sources')
    parser.add_argument('--allow-small-cpu-control',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.threads<1:parser.error('--threads must be positive')
    if not args.inspect and args.prompt is None:parser.error('--prompt is required unless --inspect is selected')
    if args.image_size is not None and args.image is None:parser.error('--image-size requires --image')
    if args.score_response is not None and (args.image is not None or not args.score_response.strip()):
        parser.error('--score-response requires a nonempty text response and the text-only human reward route')
    torch.set_num_threads(args.threads)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    expected=None if args.expected_sources is None else json.loads(Path(args.expected_sources).read_text())
    package=inspect_package(args.checkpoint,expected_checkpoint_sha256=args.expected_sha256,expected_sources=expected)
    if args.inspect:print(json.dumps(package,indent=2));return package
    model,metadata=load(package['checkpoint'],expected_checkpoint_sha256=package['checkpoint_sha256'],
        expected_sources=package['sources'],allow_mechanics=args.allow_small_cpu_control)
    tokenizer=Tokenizer.from_file(str(Path(package['checkpoint']).parent/'tokenizer.json'))
    if args.score_response is None:
        result=query(model,tokenizer,args.prompt,max_new_tokens=args.max_new_tokens,raw_prompt=args.raw_prompt,
            image=args.image,image_size=args.image_size,precision=args.precision)
    else:
        prefix,frame=prompt_ids(tokenizer,model.config,args.prompt,raw_prompt=args.raw_prompt,max_new_tokens=1)
        response=tokenizer.encode(args.score_response,add_special_tokens=False).ids
        ids=prefix[0].tolist()+response+[2]
        if not response or len(ids)>model.config.max_text_length:parser.error('The entire prompt/response/scoring-EOS must fit without truncation')
        result=dict(shared_human_reward_score=float(model.preference_score(torch.tensor([ids],dtype=torch.long))),
            complete_scoring_token_ids=ids,complete_prompt=frame,inference_precision='fp32',
            learned_human_ranking_score_not_a_program_correctness_certificate=True)
    result.update(checkpoint_sha256=package['checkpoint_sha256'],actual_registered_parameters=metadata['actual_registered_parameters'],
        trained_reward_feature_basis_preserved=True,completed_updates=package['completed_updates'],model_quality_proved=False)
    print(json.dumps(result,indent=2) if args.json or args.score_response is not None else result['response']);return result


if __name__=='__main__':main()
