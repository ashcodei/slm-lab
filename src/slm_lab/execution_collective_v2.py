"""Four-worker complete-program preference gradients and frozen own references.

No optimizer update or source admission happens here. The feedback is machine
execution on finite original publisher tests, not human preferences or RLHF.
"""
from copy import deepcopy
from dataclasses import asdict
import math

import torch
from torch import distributed as dist

from .code_residual_checkpoint_v1 import _digest
from .code_residual_collective_checkpoint_v2 import _same,_validated,fingerprint
from .corpus import file_hash
from .execution_feedback_reader_v1 import ExecutionFeedback
from .execution_preference_v1 import FORMAT,reference_scores,preference_gradients,loss
from .recurrent_ablation import state_digest

BUNDLE_FORMAT='own-complete-execution-reference-bundle-v2'


def group():
    if not dist.is_initialized() or dist.get_world_size()!=4:
        raise ValueError('Four actual initialized workers required')
    return dist.get_rank(),dist.get_world_size()


def feedback_identity(feedback,tokenizer_sha256):
    if not isinstance(feedback,ExecutionFeedback) or not _digest(tokenizer_sha256):
        raise ValueError('The admitted execution-feedback reader and exact own tokenizer required')
    feedback.guard.check()
    manifest_sha=file_hash(feedback.folder/'manifest.json')
    admission_sha=file_hash(feedback.folder/'admission.json')
    admission=feedback.admission
    if (admission.get('passed') is not True or admission.get('source_roles_only_train') is not True
            or admission.get('machine_feedback_not_human_preferences') is not True
            or admission.get('candidate_manifest_sha256')!=manifest_sha
            or feedback.manifest.get('tokenizer_sha256')!=tokenizer_sha256
            or not feedback.pairs or not feedback.positives
            or len(feedback.pairs)!=admission.get('finite_assertion_pairs')
            or len(feedback.positives)!=admission.get('complete_passing_originals')):
        raise ValueError('Every complete independently admitted pair/positive and original source role required')
    for rows in [feedback.pairs,feedback.positives]:
        if len({r['id'] for r in rows})!=len(rows) or any(r['split']!='train' or r['human_preferences'] is not False for r in rows):
            raise ValueError('All original unique source identities and train-only machine labels required')
    return dict(manifest_sha256=manifest_sha,admission_sha256=admission_sha,tokenizer_sha256=tokenizer_sha256,
        pairs=len(feedback.pairs),positives=len(feedback.positives),
        source_records_fingerprint=fingerprint(dict(pairs=feedback.pairs,positives=feedback.positives)))


def policy_identity(model,base_checkpoint_sha256,tokenizer_sha256,device,precision):
    model.assert_partition()
    if (not _digest(base_checkpoint_sha256) or not _digest(tokenizer_sha256)
            or precision not in {'fp32','bf16'}
            or any(isinstance(m,torch.nn.Dropout) and m.p for m in model.modules())):
        raise ValueError('Exact own source policy, explicit precision and deterministic forward required')
    return dict(base_checkpoint_sha256=base_checkpoint_sha256,tokenizer_sha256=tokenizer_sha256,
        frozen_base_state_sha256=state_digest(model.base.state_dict()),
        initial_residual_state_sha256=state_digest(model.residuals.state_dict()),
        config=asdict(model.config),model_counts=model.parameter_counts(),
        device_type=torch.device(device).type,precision=precision)


def prepare_reference_bundle(model,feedback,*,base_checkpoint_sha256,tokenizer_sha256,device,amp,precision):
    """Shard actual frozen reference scoring; retain every complete pair score."""
    rank,world=group()
    def prepare():
        if any(p.grad is not None for p in model.parameters()):raise ValueError('Score reference at a clean boundary')
        feedback.guard.check(full=True)
        return (policy_identity(model,base_checkpoint_sha256,tokenizer_sha256,device,precision),
            feedback_identity(feedback,tokenizer_sha256),
            [feedback.example(i) for i in range(rank,len(feedback.pairs),world)])
    policy,data,examples=_validated(prepare)
    _same(fingerprint(dict(policy=policy,feedback=data)))
    if data['pairs']<world:raise ValueError('At least one original reference pair per worker required')
    original_mode=model.training
    try:
        model.eval()
        local=_validated(lambda:reference_scores(model.code,examples,policy_sha256=fingerprint(policy),device=device,amp=amp))
    finally:
        model.train(original_mode)
    pieces=[None]*world;dist.all_gather_object(pieces,local['records'])
    def reconcile():
        records={}
        for part in pieces:
            for key,value in part.items():
                if key in records:raise ValueError('Each original reference pair must be scored exactly once')
                records[key]=value
        if set(records)!={r['id'] for r in feedback.pairs}:
            raise ValueError('Every original complete pair must have its actual frozen own reference scores')
        after=policy_identity(model,base_checkpoint_sha256,tokenizer_sha256,device,precision)
        if after!=policy:raise ValueError('Actual reference weights changed during cache construction')
        feedback.guard.check(full=True)
        cache=dict(format=FORMAT,policy_sha256=fingerprint(policy),records=records,
            response_probability='Sum over every original response/EOS token, no prompt positions.',human_preferences=False)
        return dict(format=BUNDLE_FORMAT,policy=policy,feedback=data,cache=cache,cache_fingerprint=fingerprint(cache),
            source_labels='Finite original publisher-test execution; not human feedback.',human_preferences=False)
    bundle=_validated(reconcile);_same(fingerprint(bundle))
    return bundle


def measure_preferences(model,examples,reference,*,expected_policy_sha256,alpha,beta,device,amp):
    """Collectively reject role/reference/forward errors before any gradients."""
    _,world=group()
    def preflight():
        if type(alpha) not in {int,float} or not math.isfinite(alpha) or not 0<=alpha<=1:
            raise ValueError('Explicit finite execution preference coefficient required')
        if not examples:raise ValueError('Each worker needs at least one declared pair exposure')
        # An epoch boundary may legitimately repeat a source ID. Validate and
        # score every exposure independently; never silently deduplicate draws.
        records=[]
        for example in examples:
            value=preference_gradients(model,[example],reference,expected_policy_sha256=expected_policy_sha256,
                alpha=0.,beta=beta,device=device,amp=amp)
            records.extend(value['records'])
        return dict(records=records)
    measured=_validated(preflight)
    _same(fingerprint(dict(reference=reference,policy=expected_policy_sha256,alpha=alpha,beta=beta)))
    count=torch.tensor(len(examples),device=device,dtype=torch.long);dist.all_reduce(count)
    pieces=[None]*world;dist.all_gather_object(pieces,measured['records'])
    records=[dict(rank=rank,local_exposure=i,**r) for rank,part in enumerate(pieces) for i,r in enumerate(part)]
    if len(records)!=int(count):raise ValueError('Every actual local pair must enter the global pair mean')
    for record in records:
        x=beta*record['reference_anchored_margin']
        sigmoid=math.exp(-x)/(1+math.exp(-x)) if x>=0 else 1/(1+math.exp(x))
        record['gradient_coefficient']=alpha*beta*sigmoid*world/int(count)
    summary=dict(global_pairs=int(count),unscaled_mean_DPO=sum(r['unscaled_DPO'] for r in records)/int(count),
        alpha=alpha,beta=beta,records=records,human_preferences=False)
    summary['weighted_DPO']=alpha*summary['unscaled_mean_DPO']
    return measured,summary


def backward_preferences(model,examples,measured,summary,device,amp):
    """Global pair-mean gradients; caller averages workers once after all losses."""
    _,world=group()
    def backward():
        for example,record in zip(examples,measured['records']):
            x=summary['beta']*record['reference_anchored_margin']
            sigmoid=math.exp(-x)/(1+math.exp(-x)) if x>=0 else 1/(1+math.exp(x))
            coefficient=summary['alpha']*summary['beta']*sigmoid*world/summary['global_pairs']
            if coefficient:
                (loss(model,example['chosen'],device,amp)*(coefficient*record['chosen_targets'])).backward()
                (loss(model,example['rejected'],device,amp)*(-coefficient*record['rejected_targets'])).backward()
    _validated(backward)


def execution_gradients(model,examples,reference,*,expected_policy_sha256,alpha,beta,device,amp):
    measured,summary=measure_preferences(model,examples,reference,expected_policy_sha256=expected_policy_sha256,
        alpha=alpha,beta=beta,device=device,amp=amp)
    backward_preferences(model,examples,measured,summary,device,amp)
    return deepcopy(summary)
