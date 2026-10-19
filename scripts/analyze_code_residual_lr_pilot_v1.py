"""Recompute pilot source losses and descriptive problem-group sensitivity."""
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json


ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'reports/code-residual-lr-pilot-v1';output=ROOT/'reports/code-residual-lr-pilot-v1-source-learning-analysis.json'
    if output.exists():raise ValueError('Preserve earlier source-learning analysis')
    result=json.loads((folder/'results.json').read_text());plan=json.loads((folder/'plan.json').read_text())
    selection=json.loads((folder/'selected-original-rows.json').read_text())
    baseline=json.loads((folder/'initial-source-NLL.json').read_text())
    if file_hash(folder/'plan.json')!=result['run_plan_sha256'] or file_hash(folder/'selected-original-rows.json')!=plan['selection_sha256']:
        raise ValueError('Exact immutable original source selection and completed pilot plan required')
    profiles=[json.loads((folder/(p['name']+'.json')).read_text()) for p in plan['profiles']]
    by_id={r['id']:r for role in selection for r in selection[role]}
    groups=sorted({r['group_id'] for r in selection['validation']});positions={g:i for i,g in enumerate(groups)}
    def source_metric(metric,role):
        rows=metric['per_row'];expected=selection[role]
        if len(rows)!=len(expected):raise ValueError('Every original source row required')
        for row,e in zip(rows,expected):
            if (row['id']!=e['id'] or row['role']!=role or row['language']!=e['language']
                    or row['targets']!=e['target_tokens'] or not np.isfinite(row['mean_nll'])):
                raise ValueError('Original programme/source targets and finite per-row loss required')
        targets=sum(r['targets'] for r in rows);weighted=sum(r['targets']*r['mean_nll'] for r in rows)/targets
        if metric['response_targets']!=targets or abs(metric['token_mean_nll']-weighted)>1e-12:
            raise ValueError('Full target-token-normalized source aggregate must rederive')
        language=defaultdict(lambda:[0.,0,0]);group_loss=np.zeros(len(groups));group_targets=np.zeros(len(groups))
        for row in rows:
            bucket=language[row['language']];bucket[0]+=row['targets']*row['mean_nll'];bucket[1]+=row['targets'];bucket[2]+=1
            if role=='validation':
                i=positions[by_id[row['id']]['group_id']];group_loss[i]+=row['targets']*row['mean_nll'];group_targets[i]+=row['targets']
        return weighted,{k:dict(token_mean_nll=v[0]/v[1],targets=v[1],programmes=v[2]) for k,v in sorted(language.items())},group_loss,group_targets
    initial_train,_,_,_=source_metric(baseline['train'],'train')
    initial_valid,initial_language,initial_groups,weights=source_metric(baseline['validation'],'validation')
    draws=np.random.default_rng(910453).integers(0,len(groups),size=(5000,len(groups)))
    descriptions=[];group_values=[]
    for profile in profiles:
        train,_,_,_=source_metric(profile['train'],'train')
        valid,languages,values,current_weights=source_metric(profile['validation'],'validation')
        if not np.array_equal(weights,current_weights):raise ValueError('Paired original validation group counts must match')
        delta=values-initial_groups
        resampled=delta[draws].sum(1)/weights[draws].sum(1)
        descriptions.append(dict(profile=profile['profile'],train_token_mean_nll=train,validation_token_mean_nll=valid,
            relative_validation_NLL_reduction=1-valid/initial_valid,
            paired_validation_NLL_change=valid-initial_valid,
            paired_problem_group_resampling_95_percent_range=np.quantile(resampled,[.025,.975]).tolist(),
            validation_language_losses=languages,
            validation_language_labels_with_lower_NLL=sum(v['token_mean_nll']<initial_language[k]['token_mean_nll'] for k,v in languages.items())))
        group_values.append(values)
    winner=min(range(len(descriptions)),key=lambda i:descriptions[i]['validation_token_mean_nll'])
    if descriptions[winner]['profile']!=result['best_profile']:raise ValueError('Predeclared internal-validation selection must reproduce')
    warmup_delta=(group_values[1]-group_values[0])[draws].sum(1)/weights[draws].sum(1)
    report=dict(completed_at=datetime.now(timezone.utc).isoformat(),source_plan_sha256=result['run_plan_sha256'],
        completed_pilot_results_sha256=file_hash(folder/'results.json'),selection_sha256=plan['selection_sha256'],
        original_train_programmes=512,original_validation_programmes=128,validation_problem_groups=len(groups),
        original_train_targets=baseline['train']['response_targets'],original_validation_targets=baseline['validation']['response_targets'],
        initial_train_token_mean_nll=initial_train,initial_validation_token_mean_nll=initial_valid,
        initial_validation_language_losses=initial_language,profiles=descriptions,selected_profile=result['best_profile'],
        equal_peak_warmup_minus_constant_validation_NLL=descriptions[1]['validation_token_mean_nll']-descriptions[0]['validation_token_mean_nll'],
        equal_peak_warmup_minus_constant_paired_group_resampling_95_percent_range=np.quantile(warmup_delta,[.025,.975]).tolist(),
        selected_without_functional_or_QA_test_feedback=True,functional_gain_inferred_from_teacher_NLL=False,
        candidate_promoted=False,goal_achieved=False,
        scope='Descriptive paired-group sensitivity of this stratified128-programme/117-problem internal-validation sample, including deliberately selected size extremes. Resampling is not a population/universal-capability confidence claim or an independent initialization trial. Teacher-forced source prediction does not establish generated-program correctness or all-round quality.')
    atomic_json(output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['profiles','initial_validation_language_losses']},indent=2))
    print('profiles',json.dumps([{k:v for k,v in p.items() if k!='validation_language_losses'} for p in descriptions],indent=2))


if __name__=='__main__':main()
