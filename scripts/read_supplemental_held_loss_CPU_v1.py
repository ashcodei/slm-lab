"""Rebuild saved held-loss populations, masks and stratified arithmetic.

Reads the actual immutable native token store directly, without importing
the training/evaluation producers or loading model/optimizer state. Neural
likelihoods are not rerun; this is source/metric readback, not a fresh test.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/tmp/slm-supplemental-public-code-full-GPU-v1')
DATA=ROOT/'data/public-reasoned-code-SFT-v1'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def close(a,b):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)


def main(endpoints,output):
    if output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'torch' in sys.modules:
        raise ValueError('A fresh explicitly CPU-only source/metric observation is required')
    if not endpoints or len(set(endpoints))!=len(endpoints):raise ValueError('Distinct completed held endpoints required')
    os.sched_setaffinity(0,{22});os.nice(10)
    import numpy as np
    plan=read(RUN/'plan.json');manifest=read(DATA/'manifest.json')
    if (sha(DATA/'manifest.json')!=plan['data_manifest_sha256']
            or manifest['format']!='slm-codenet-complete-token-pack-v1'
            or manifest['counts']['validation']['examples']!=425):
        raise ValueError('The exact original complete reserved native pack is required')
    files={str(p):sha(p) for p in [Path(__file__).resolve(),RUN/'plan.json',DATA/'manifest.json']}
    for name,digest in manifest['files_sha256'].items():
        p=DATA/name
        if sha(p)!=digest:raise ValueError('An original native pack byte changed')
        files[str(p)]=digest
    index=np.load(DATA/'index.npy',allow_pickle=False)
    expected=np.dtype([('offset','<u8'),('length','<u2'),('target_start','<u2'),('role','u1')])
    if index.dtype!=expected or index.ndim!=1 or len(index)!=4100:
        raise ValueError('All original native row indexes and their exact schema are required')
    with gzip.open(DATA/'original-rows.jsonl.gz','rt') as f:rows=[json.loads(line) for line in f]
    raw=(DATA/'input-ids.bin').read_bytes();offset=0;held=[]
    if len(rows)!=len(index):raise ValueError('Every original complete source row is required')
    for position,item in enumerate(index):
        n,start=int(item['length']),int(item['target_start']);role=int(item['role']);row=rows[position]
        if (int(item['offset'])!=offset or not 3<=n<=4096 or not 1<=start<n-1 or role not in [0,1]
                or row['token_pack_index']!=position or row['view'] not in ['code_only','reasoning']
                or row['split']!=('train' if role==0 else 'validation')):
            raise ValueError('The exact original complete native offsets, masks, views and roles are required')
        values=list(struct.unpack_from('<'+'H'*n,raw,offset*2));offset+=n
        if values[0]!=1 or values[-1]!=2 or any(not 4<=v<32768 for v in values[1:-1]):
            raise ValueError('Every original full native input/BOS/EOS must remain valid')
        if role==1:
            if row['original_fixed_role']['source_role']!='reserved':raise ValueError('Original reserved source role changed')
            held.append(dict(pack_position=position,view=row['view'],ids=values,labels=[-100]*start+values[start:],targets=n-start))
    if offset*2!=len(raw) or len(held)!=425 or sum(h['targets'] for h in held)!=367945:
        raise ValueError('The entire original reserved source population and target budget are required')
    results={}
    for endpoint in ['baseline',*endpoints]:
        path=RUN/endpoint/'assessment/held-code-NLL.json';report=read(path);files[str(path)]=sha(path)
        records=report['records']
        if (not report['complete'] or report['reserved_views']!=425 or len(records)!=425
                or report['reserved_optimization_updates']!=0 or not report['all_original_masks_and_targets_preserved']):
            raise ValueError('Every actual held record and evaluation-only budget must remain complete')
        grouped={view:dict(rows=0,targets=0,CE=[],EOS=[]) for view in ['all','code_only','reasoning']}
        for position,(saved,original) in enumerate(zip(records,held)):
            if (saved['reserved_local_position']!=position or saved['original_pack_position']!=original['pack_position']
                    or saved['complete_input_token_ids']!=original['ids']
                    or saved['original_response_EOS_labels']!=original['labels']
                    or saved['response_EOS_targets']!=original['targets']):
                raise ValueError('Every saved original prompt, response, mask and final EOS must reconstruct exactly')
            ce,eos=saved['CE_sum'],saved['EOS_CE_sum']
            if (any(type(v) not in [int,float] or not math.isfinite(v) or v<0 for v in [ce,eos]) or eos>ce+1e-5):
                raise ValueError('Complete original finite nonnegative token/EOS loss sums required')
            for view in ['all',original['view']]:
                g=grouped[view];g['rows']+=1;g['targets']+=original['targets'];g['CE'].append(ce);g['EOS'].append(eos)
        summary={view:dict(views=g['rows'],response_EOS_targets=g['targets'],
            token_mean_NLL=math.fsum(g['CE'])/g['targets'],EOS_token_mean_NLL=math.fsum(g['EOS'])/g['rows'])
            for view,g in grouped.items()}
        if (report['response_EOS_targets']!=summary['all']['response_EOS_targets']
                or not close(report['token_mean_NLL'],summary['all']['token_mean_NLL'])
                or not close(report['EOS_token_mean_NLL'],summary['all']['EOS_token_mean_NLL'])):
            raise ValueError('All original aggregate denominators and loss arithmetic must reconstruct')
        results[endpoint]=summary
    comparisons={}
    for endpoint in endpoints:
        comparisons[endpoint]={view:dict(token_NLL_relative_reduction_percent=100*(1-result['token_mean_NLL']/results['baseline'][view]['token_mean_NLL']),
            EOS_NLL_relative_reduction_percent=100*(1-result['EOS_token_mean_NLL']/results['baseline'][view]['EOS_token_mean_NLL']))
            for view,result in results[endpoint].items()}
    if any(sha(p)!=h for p,h in files.items()):raise ValueError('A consumed source/metric/runtime byte changed')
    result=dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        all_original_native_rows_read=4100,all_reserved_views=425,reserved_response_EOS_targets=367945,
        all_original_masks_and_source_views_independently_reconstructed=True,results=results,comparisons=comparisons,
        same_original_input_and_target_ids_across_endpoints=True,neural_likelihoods_or_model_weights_replayed=False,
        reused_development_data_not_fresh_independent_test=True,generated_code_correctness_improvement_proved=False,
        no_GPU_or_model_or_optimizer_loaded=True,candidate_promoted=False,goal_achieved=False)
    with output.open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({k:result[k] for k in ['complete','results','comparisons','neural_likelihoods_or_model_weights_replayed']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint',required=True,action='append',choices=['plain_tokens','eos_weight4'])
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args();main(args.endpoint,args.output)
