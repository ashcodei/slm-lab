"""Admission of authentic historical source targets, with explicit qualifications.

This policy admits original published Accepted programmes as supervised text.
It does not turn historical judge labels into current execution certificates,
synthetic preference labels, or claims about a trained model's correctness.
"""
from collections import Counter
import hashlib

import numpy as np

from .codenet_token_pack_v1 import ROLES


POLICY = 'historical-accepted-source-SFT-with-explicit-runtime-qualifications-v1'


def audit_source_rows(pack, rows, encoder, special_ids, qualified_native, held_references):
    if not {encoder.bos, encoder.eos}.issubset(special_ids):
        raise ValueError('Exact original special-token framing required')
    held_groups={r['group_id'] for r in held_references}
    held_bodies={r['original_source_body_sha256'] for r in held_references}
    group_roles,body_roles,pairs,seen,native_seen={}, {}, set(),set(),set()
    counts,language_counts=Counter(),Counter()
    reverse={v:k for k,v in ROLES.items()}
    longest=0
    for position,row in enumerate(rows):
        if position>=len(pack.index):raise ValueError('Extra original source row')
        item=pack.index[position];role=row['split'];member=row['original_archive_member']
        if (member in seen or row['token_pack_index']!=position or reverse[int(item['role'])]!=role
                or row.get('language')=='Text' or row.get('image')
                or row.get('license')!='CDLA-Permissive-2.0'
                or row.get('original_judge_status')!='Accepted'
                or row.get('original_submission_metadata',{}).get('status')!='Accepted'):
            raise ValueError('Unique original licensed Accepted text source with fixed role required')
        seen.add(member)
        if not row['response'].strip() or '\0' in row['response']:
            raise ValueError('Original nonempty programme text required')
        body=hashlib.sha256(row['response'].encode()).hexdigest()
        if body!=row['original_source_body_sha256']:
            raise ValueError('Original programme body bytes differ')
        for mapping,key in [(group_roles,row['group_id']),(body_roles,body)]:
            if key in mapping and mapping[key]!=role:
                raise ValueError('Original programme or component crosses sealed roles')
            mapping[key]=role
        if role=='train' and (body in held_bodies or row['group_id'] in held_groups):
            raise ValueError('Held algorithm reference/component cannot enter training')
        ids,labels=encoder.sequence(row,'response')
        start=next(i for i,v in enumerate(labels) if v!=-100)
        offset,length,target_start=int(item['offset']),int(item['length']),int(item['target_start'])
        stored=pack.ids[offset:offset+length]
        digest=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()
        if (not np.array_equal(stored,np.asarray(ids,dtype='<u2')) or start!=target_start
                or labels!=[-100]*start+ids[start:] or digest!=row['complete_input_ids_sha256']
                or row['target_tokens']!=len(ids)-start):
            raise ValueError('Every complete original prompt/program/EOS token and mask must reconstruct')
        if any(v in special_ids for v in ids[1:-1]):
            raise ValueError('Literal internal reserved model tokens require separate source review')
        pair=(digest,start)
        if pair in pairs:raise ValueError('Previously unique complete source target repeated')
        pairs.add(pair)
        contract=row.get('execution_contract')
        if member in qualified_native:
            q=qualified_native[member]
            if (any(row[k]!=q[k] for k in ['original_source_body_sha256','source_problem_id','group_id','split'])
                    or not isinstance(contract,dict) or contract.get('kind')!='source-declared-numba-aot-v1'
                    or contract.get('preparation_final_argument')!='ONLINE_JUDGE'
                    or contract.get('generated_module')!='my_module'
                    or contract.get('actual_native_preparation_verified') is not False
                    or contract.get('original_response_modified') is not False):
                raise ValueError('Exact source-declared native contract and unverified-execution qualification required')
            native_seen.add(member)
        elif contract is not None:
            raise ValueError('Undeclared runtime contract requires independent review')
        for key,value in [('examples',1),('context_tokens',len(ids)),('target_tokens',len(ids)-start)]:
            counts[role,key]+=value
        language_counts[row['language']]+=1;longest=max(longest,len(ids))
    actual={role:{k:counts[role,k] for k in ['examples','context_tokens','target_tokens']} for role in ROLES}
    if (len(seen)!=pack.manifest['source_examples'] or actual!=pack.manifest['counts']
            or native_seen!=set(qualified_native) or encoder.truncated):
        raise ValueError('Complete unchanged source population, native declarations and full context fit required')
    return dict(policy=POLICY,actual_counts=actual,original_examples=len(seen),
        training_examples=actual['train']['examples'],language_counts=dict(language_counts),
        distinct_programme_body_hashes=len(body_roles),cross_role_exact_programme_bodies=0,
        cross_role_components=0,known_held_algorithm_training_body_or_component_overlaps=0,
        literal_internal_reserved_model_tokens=0,longest_complete_context=longest,
        every_original_prompt_programme_and_full_target_mask_reencoded=True,
        qualified_native_contract_sources=sorted(native_seen),actual_native_preparation_executed=False,
        execution_quality_selection_or_preference_labels_created=False,
        scope='Original historical Accepted-source supervised text, retaining runtime/quality qualifications. No universal execution, hidden-test correctness, global contamination exclusion or learned-capability certificate.')
