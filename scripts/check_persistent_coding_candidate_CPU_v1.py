"""Query a relocated complete policy without scratch/training/ancestor reads.

The eight-action probe checks delivery equivalence, not answer correctness.
An explicit fresh output path preserves all previous observations.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'checkpoints/public-coding-research-v1'
FORBIDDEN=['/tmp/slm-supplemental-public-code-full-GPU-v1',
    '/tmp/slm-completed-public-coding-standalone-CPU-v1','/tmp/slm-completed-joint-initializer-CPU-v3']


def main(profile,output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or output.exists():
        raise ValueError('An explicitly CPU-only fresh delivery observation is required')
    os.sched_setaffinity(0,{22});os.nice(10)
    folder=DEST/profile;proof=json.loads((folder/'complete.json').read_text())
    preserved=json.loads((folder/'PRESERVED.json').read_text())
    if (not proof['complete'] or not preserved['complete'] or proof['completed_updates']!=230
            or preserved['checkpoint_sha256']!=proof['checkpoint_sha256']):
        raise ValueError('The exact complete physically preserved policy is required')
    blocked=[];network=[]
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=os.path.abspath(os.fsdecode(args[0]))
            if any(path==p or path.startswith(p+'/') for p in FORBIDDEN):
                blocked.append(path);raise RuntimeError('Inference cannot read scratch/training/ancestor files')
        if event in ['socket.connect','socket.getaddrinfo']:
            network.append(event);raise RuntimeError('Standalone inference cannot call network services')
    sys.addaudithook(audit)
    from slm_lab.own_coding_policy_standalone_v1 import main as query
    sys.argv=['own_coding_policy_standalone_v1','--checkpoint',str(folder/'model.pt'),'--prompt',
        'Write a Python function to add two integers.','--max-new-tokens','8','--precision','fp32','--threads','1','--json']
    buffer=io.StringIO()
    with contextlib.redirect_stdout(buffer):query()
    result=json.loads(buffer.getvalue())
    if (result['actual_generated_token_ids']!=proof['outputs']['text_actions']
            or result['checkpoint_sha256']!=proof['checkpoint_sha256']
            or result['actual_registered_parameters']!=999706442 or result['completed_checkpoint_updates']!=230
            or blocked or network):
        raise ValueError('Actual persistent inference must match every original source action and complete identity')
    with Path(__file__).open('rb') as f:implementation=hashlib.file_digest(f,'sha256').hexdigest()
    report=dict(complete=True,profile=profile,implementation_sha256=implementation,
        persistent_model_and_tokenizer_only=True,forbidden_original_paths=FORBIDDEN,
        blocked_source_read_attempts=blocked,network_attempts=network,actual_query=result,
        actual_actions_equal_original=True,completed_epoch_updates=230,query_is_not_quality_assessment=True,
        candidate_promoted=False,goal_achieved=False)
    with output.open('x') as f:
        f.write(json.dumps(report,indent=2)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(complete=True,profile=profile,completed_epoch_updates=230,
        actual_actions_equal_original=True,blocked_source_read_attempts=blocked,network_attempts=network,
        query_is_not_quality_assessment=True,candidate_promoted=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',required=True,choices=['plain_tokens','eos_weight4'])
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    main(args.profile,args.output)
