"""Evaluation-only original valid-stream next-token loss, with no optimizer."""
from contextlib import nullcontext

import torch
from torch import distributed as dist


def evaluate(model,packed,*,device='cpu',rank=0,world=1,maximum_windows=128,amp=None):
    if type(maximum_windows) is not int or maximum_windows<1 or type(world) is not int or world<1 or not 0<=rank<world:
        raise ValueError('Explicit bounded complete held window population required')
    family=packed.families[0];population=packed.size(family,split='valid');count=min(maximum_windows,population)
    positions=[i*population//count for i in range(count)];targets=0;loss=0.;model.eval();amp=amp or nullcontext
    with torch.no_grad():
        for position in positions[rank::world]:
            row=packed.row(family,position,split='valid')
            if row.get('split')!='valid' or row.get('source_split')!='valid':raise ValueError('Only original held source windows may supply validation loss')
            ids=torch.tensor([row['tokens']],device=device);n=len(row['tokens'])-1
            with amp():mean=model(input_ids=ids,labels=ids,attention_mask=torch.ones_like(ids,dtype=torch.bool))['loss']
            if not bool(torch.isfinite(mean)):raise ValueError('Held complete-source loss must be finite')
            loss+=float(mean)*n;targets+=n
    values=torch.tensor([loss,targets],dtype=torch.float64,device=device)
    if world>1:dist.all_reduce(values)
    total_loss,total_targets=values.cpu().tolist()
    return dict(original_split='valid',selected_original_positions=positions,original_valid_population=population,
        evaluated_windows=count,complete_causal_targets=int(total_targets),token_mean_NLL=total_loss/total_targets,
        maximum_original_context=packed.manifest['maximum_length'],no_optimizer_or_gradient=True,
        original_manifest_sha256=packed.manifest['packed_manifest_sha256'],
        full_original_valid_stream=count==population,not_an_independent_native_code_correctness_or_all_domain_score=True,goal_achieved=False)
