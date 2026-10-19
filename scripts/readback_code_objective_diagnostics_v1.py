"""Reconcile all saved per-target predictions and raw published-sample executions."""
import argparse
import base64
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import string

import numpy as np
from tokenizers import Tokenizer

from slm_lab.codenet_sample_oracles_v1 import SUPPORTED,check_sample
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
CATEGORIES=['true_EOS','whitespace_only_piece','ASCII_punctuation_only_piece','digit_only_piece','other_text_piece']


def read(p):return json.loads(p.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['CE_control','CE_plus_repetition_UL'],required=True)
    args=parser.parse_args();folder=ROOT/'reports/code-repetition-objective-token-diagnostics-v1'/args.profile
    output=folder/'independent-prediction-and-sample-readback.json'
    if output.exists():raise ValueError('Preserve any previous diagnostic readback')
    plan=read(folder/'plan.json');result=read(folder/'results.json')
    assert all(file_hash(Path(p))==h for p,h in plan['source_files_sha256'].items())
    parent=ROOT/'reports/code-repetition-objective-pilot-v1';parent_plan=read(parent/'plan.json')
    assert all(file_hash(Path(p))==h for p,h in parent_plan['source_files_sha256'].items())
    assert file_hash(folder/'records.json')==result['records_sha256']
    assert file_hash(folder/'per-target-predictions.npz')==result['targets_sha256']
    assert result['policy_state_sha256_before']==result['policy_state_sha256_after']
    records=read(folder/'records.json')['records']
    flat=np.load(folder/'per-target-predictions.npz',allow_pickle=False)['targets']
    assert flat.dtype==np.dtype([('gold','<u2'),('prediction','<u2'),('NLL','<f4'),('category','u1')])
    assert len(flat)==result['complete_target_tokens']==414356
    assert bool(np.isfinite(flat['NLL']).all()) and bool((flat['NLL']>=0).all())
    assert bool((flat['prediction']<32768).all())
    selection=read(ROOT/'reports/code-residual-lr-pilot-v1/selected-original-rows.json')
    rows=selection['train']+selection['validation'];assert len(records)==len(rows)==640
    packs={role:CompleteCodePack(ROOT/'data/codenet-durable-pack-v2',role,purpose='audit') for role in ['train','validation']}
    token=Tokenizer.from_file(str(ROOT/'checkpoints/code-residual-lr-pilot-v1/tokenizer.json'))
    category=[]
    for value in range(32768):
        decoded=token.decode([value],skip_special_tokens=False);text=decoded.strip()
        if value==2:k=0
        elif decoded.isspace():k=1
        elif text and set(text)<=set(string.punctuation)-{'_'}:k=2
        elif text.isdigit():k=3
        else:k=4
        category.append(k)
    category=np.array(category,dtype=np.uint8);aggregates=defaultdict(lambda:dict(tokens=0,NLL_sum=0.,top1_matches=0))
    by_role=defaultdict(lambda:dict(programs=0,targets=0,top1_matches=0,teacher_all_reference_targets_argmax=0,
        first_response_top1_matches=0,true_EOS_top1_matches=0))
    offset=0
    for observed,row in zip(records,rows):
        assert observed['id']==row['id'] and observed['role']==row['role']
        ids,labels=packs[row['role']].sequence(row['local_position']);start=next(i for i,y in enumerate(labels) if y!=-100)
        assert hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()==row['full_input_ids_sha256']
        assert observed['offset']==offset and observed['target_tokens']==row['target_tokens']
        data=flat[offset:offset+row['target_tokens']];offset+=len(data)
        assert np.array_equal(data['gold'],ids[start:]) and np.array_equal(data['category'],category[data['gold']])
        assert int(data['gold'][-1])==2
        for i,name in enumerate(CATEGORIES):
            part=data[data['category']==i]
            expected=dict(tokens=len(part),NLL_sum=float(part['NLL'].astype(np.float64).sum()),
                top1_matches=int((part['gold']==part['prediction']).sum()))
            assert expected==observed['categories'][name]
            for key,value in expected.items():aggregates[(row['role'],name)][key]+=value
        for name,index in [('first_response',0),('true_EOS',-1)]:
            boundary=observed['boundaries'][name]
            assert boundary['gold']==int(data['gold'][index]) and boundary['prediction']==int(data['prediction'][index])
            assert abs(boundary['NLL']-float(data['NLL'][index]))<1e-6
        if 'sealed_free_generation' in observed:
            free=observed['sealed_free_generation'];p=free['matching_prefix']
            if p<len(data):
                assert abs(free['first_divergence_teacher_gold_NLL']-float(data['NLL'][p]))<1e-6
                assert free['first_divergence_teacher_argmax']==int(data['prediction'][p])
        totals=by_role[row['role']];totals['programs']+=1;totals['targets']+=len(data)
        matches=data['gold']==data['prediction'];totals['top1_matches']+=int(matches.sum())
        totals['teacher_all_reference_targets_argmax']+=int(matches.all())
        totals['first_response_top1_matches']+=int(matches[0]);totals['true_EOS_top1_matches']+=int(matches[-1])
    assert offset==len(flat)
    summaries={role:{name:dict(values,mean_NLL=values['NLL_sum']/values['tokens'] if values['tokens'] else None,
        top1_accuracy=values['top1_matches']/values['tokens'] if values['tokens'] else None)
        for (r,name),values in aggregates.items() if r==role} for role in ['train','validation']}
    assert summaries==result['literal_piece_categories']
    execution=parent/f'{args.profile}-published-sample-controls'
    execution_plan=read(execution/'plan.json');execution_result=read(execution/'results.json')
    assert file_hash(execution/'plan.json')==execution_result['plan_sha256']
    assert file_hash(execution/'records.json')==execution_result['records_sha256']
    assert all(file_hash(Path(p))==h for p,h in execution_plan['source_files_sha256'].items())
    adapter=read(execution/'path-adapter-provenance.json')
    assert file_hash(Path(adapter['adapter']))==adapter['adapter_sha256']
    assert file_hash(Path(adapter['original_driver']))==adapter['original_driver_sha256']
    descriptions={r['problem_id']:r for r in map(json.loads,(ROOT/'reports/codenet-description-format-v1/records.jsonl').open())}
    runs=read(execution/'records.json')['records'];assert len(runs)==4
    executed=0;defined=0
    for run in runs:
        pairs=descriptions[run['source_problem_id']]['samples'];defined+=len(pairs)
        for sample,pair in zip(run['samples'],pairs):
            raw=base64.b64decode(sample['stdout_base64'],validate=True)
            assert hashlib.sha256(raw).hexdigest()==sample['stdout_sha256']
            oracle=check_sample(run['source_problem_id'],pair['input'],pair['output'],raw.decode(errors='replace')) if sample['status']=='completed' and run['source_problem_id'] in SUPPORTED else None
            assert oracle==sample['source_specific_oracle']
            agrees=bool(sample['status']=='completed' and (oracle['passed'] if oracle is not None else raw.split()==pair['output'].encode().split()))
            assert sample['sample_agrees']==agrees;executed+=1
        if run['samples']:
            assert len(run['samples'])==len(pairs)
            assert run['qualification']==('all_original_published_samples_agree' if all(s['sample_agrees'] for s in run['samples'])
                else 'one_or_more_original_published_samples_disagree_or_runtime_failure')
    expected={role:dict(programs=2,all_published_samples_agree=sum(r['qualification']=='all_original_published_samples_agree' for r in runs if r['role']==role)) for role in ['train','validation']}
    assert execution_result['summary']==expected and defined==11
    atomic_json(output,dict(verified_at=datetime.now(timezone.utc).isoformat(),profile=args.profile,
        original_sources=640,complete_per_target_gold_prediction_category_and_NLL_records_rederived=len(flat),
        teacher_prediction_summary=dict(by_role),literal_piece_categories=summaries,
        published_sample_controls=expected,original_published_sample_cases_defined=defined,
        sample_cases_actually_executed=executed,case_runs_blocked_by_compilation=defined-executed,
        model_updates=0,primary_benchmark_modified=False,candidate_promoted=False,goal_achieved=False,
        scope='Saved predictions/aggregates and published stdout/oracle readback, not an independent model rerun. Teacher-forced exact-reference predictions are not free-generation/functional success; other valid programs remain possible. Boundary ranks were observed during inference and are not rederived without saved full logits.'))
    print(json.dumps(dict(verified=True,profile=args.profile,teacher_summary=dict(by_role),published_samples=expected)))


if __name__=='__main__':main()
