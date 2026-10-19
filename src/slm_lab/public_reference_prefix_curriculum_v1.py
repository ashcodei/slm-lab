"""Training-only partial-program views using the exact original human reference.

This is a proposed reward curriculum, not an unassisted capability metric. Each
view moves an unchanged original response prefix into the rollout context; its
remaining native target, complete original CE anchor and source identity stay
explicit. A separately executed prefix-only control must reject trivial EOS
solutions before any such view is admitted to an RL experiment.
"""
import hashlib

import numpy as np


QUARTERS=(3,2,1,0)


def view(tokenizer, row, ids, labels, retained_quarters):
    if (row.get('split')!='train' or row.get('publisher_split')!='train'
            or row.get('external_model_teacher_used') is not False or row.get('model_generated_examples') is not False):
        raise ValueError('Only explicit original public human training solutions may provide a curriculum prefix')
    if type(retained_quarters) is not int or retained_quarters not in QUARTERS:
        raise ValueError('Explicit fixed3/2/1/0-quarter curriculum level required')
    if (not 3<=len(ids)<=4096 or len(ids)!=len(labels) or ids[0]!=1 or ids[-1]!=2
            or any(type(v) is not int or not 0<=v<32768 for v in ids)):
        raise ValueError('Whole original native source sequence required')
    target_start=next((i for i,v in enumerate(labels) if v!=-100),len(labels))
    if not 1<=target_start<len(ids)-1 or labels[:target_start]!=[-100]*target_start or labels[target_start:]!=ids[target_start:]:
        raise ValueError('All original contiguous response/EOS targets required')
    digest=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()
    if digest!=row['complete_input_ids_sha256']:raise ValueError('Original source native bytes differ')
    reference=ids[target_start:-1];response=row['response']
    encoded=tokenizer.encode(response,add_special_tokens=False)
    if encoded.ids!=reference or tokenizer.decode(reference,skip_special_tokens=False)!=response:
        raise ValueError('Every original public response byte must match its exact native response tokens')
    # Retain a complete line ending at an original token boundary. At least16
    # original non-EOS response tokens remain whenever any hint is supplied.
    maximum=min(len(reference)*retained_quarters//4,max(0,len(reference)-16))
    keep=0
    if retained_quarters:
        for position,(_,end) in enumerate(encoded.offsets[:maximum],1):
            if end and response[end-1]=='\n':keep=position
    prefix_text=tokenizer.decode(reference[:keep],skip_special_tokens=False)
    if prefix_text!=response[:len(prefix_text)] or keep and not prefix_text.endswith('\n'):
        raise ValueError('A complete original line prefix is required; no synthetic code may be supplied')
    native_prefix=ids[:target_start+keep];suffix=ids[target_start+keep:]
    if native_prefix+suffix!=ids or suffix[-1]!=2 or keep and len(suffix)-1<16:
        raise ValueError('The original full program and genuine EOS must reconstruct without clipping')
    budget=min(4096-len(native_prefix),max(512,len(suffix)+64))
    if budget<len(suffix):raise ValueError('Every complete original missing suffix must fit the actual rollout budget')
    return dict(id=row['id'],retained_reference_quarters=retained_quarters,
        original_native_input_sha256=digest,original_native_context_positions=len(ids),
        original_response_EOS_targets=len(ids)-target_start,
        original_native_prompt_ids=ids[:target_start],original_response_start=target_start,
        native_rollout_prefix_ids=native_prefix,public_reference_prefix_ids=reference[:keep],
        public_reference_prefix_text=prefix_text,retained_reference_tokens=keep,
        original_missing_suffix_and_EOS_ids=suffix,missing_original_response_EOS_targets=len(suffix),
        complete_original_anchor_ids=ids,complete_original_anchor_labels=labels,
        complete_missing_suffix_training_budget=budget,
        exact_unassisted_original_source=keep==0,training_only=True,unassisted_capability_score=False,
        prefix_only_execution_control_required_before_RL_admission=True,
        prefix_only_execution_control_passed=False,admitted_to_RL=False,
        model_updates=0,goal_achieved=False)
