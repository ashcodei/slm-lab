"""Full original source-shaped checkpoint mappings, with no neural updates."""
import copy
import json
from pathlib import Path

import pytest

from slm_lab.complete_joint_objective_bridge_v1 import checkpoint_entry
from slm_lab.complete_joint_source_schedule_v1 import digest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def original():
    streams={name:json.loads((ROOT/'reports/complete-joint-source-schedule-CPU-v1'/(name+'.json')).read_text()) for name in ['code','human','visual']}
    joint=json.loads((ROOT/'reports/spread-joint-source-schedule-CPU-v1/joint.json').read_text())
    def header(row):
        parts={}
        for reference in row['components']:
            name=reference['stream'];entry=streams[name]['entries'][reference['source_step']-1]
            assert digest(entry)==reference['original_entry_sha256']
            parts[name]=dict(reference=reference,entry=entry)
        return dict(step=row['step'],source_entry_sha256=digest(row),components=parts)
    return [header(row) for row in joint['entries']]


def test_every_original_joint_entry_retains_whole_targets_sparse_tasks_and_real_weighted_denominators(original):
    assert len(original)==5362
    contexts=targets=bytes_=words=negative=correction=0;code=human=photo=0
    for metadata in original:
        entry=checkpoint_entry(metadata,source_identity_sha256='a'*64)
        assert entry['step']==metadata['step']
        seen=[]
        for row in entry['objectives']:
            assert row['source_sha256']=='a'*64 and row['source_role']=='train'
            seen.append(row['task']);contexts+=row['complete_context_tokens'];targets+=row['response_EOS_targets']
            bytes_+=row['word_UTF8_bytes'];denominators=row['normalization_denominators']
            if 'original_positive_weighted_targets' in denominators:
                code+=1
                assert denominators['original_positive_weighted_targets']==metadata['components']['code']['entry']['original_global_positive_weighted_targets']
            if 'human_pairs' in denominators:
                human+=denominators['human_pairs']
                assert denominators['chosen_response_EOS']==metadata['components']['human']['entry']['chosen_response_EOS_targets']
            if 'whole_words' in denominators:words+=denominators['whole_words']
            if 'image_pairs' in denominators:photo+=denominators['image_pairs']
            negative+=denominators.get('negative_EOS',0);correction+=denominators.get('corrective_pairs',0)
        if 'visual' in metadata['components'] and metadata['components']['visual']['entry']['family']=='words':
            assert seen[-2:]==['native_image','word_CTC']
    assert (contexts,targets)==(42271575+31147346+17681556,21316734+8514104+4669243)
    assert (bytes_,words,negative,correction,code,human,photo)==(644063,91585,80,4,5256,85786,78255)


@pytest.mark.parametrize('fault',['weighted_mass','chosen_context','rejected_targets','word_bytes','photo_negative','correction_negative'])
def test_dropped_or_conflated_original_target_denominators_reject(original,fault):
    if fault=='weighted_mass':
        metadata=copy.deepcopy(original[0]);metadata['components']['code']['entry']['original_global_positive_weighted_targets']=float('nan')
    elif fault in ['chosen_context','rejected_targets']:
        metadata=copy.deepcopy(original[0]);key='chosen_context_tokens' if fault=='chosen_context' else 'rejected_response_EOS_targets'
        metadata['components']['human']['entry'][key]+=1
    else:
        family='words' if fault=='word_bytes' else 'images:additional_photo_captions'
        if fault=='correction_negative':
            source=next(m for m in original if 'visual' in m['components'] and m['components']['visual']['entry']['complete_original_correction_indices'])
        else:source=next(m for m in original if 'visual' in m['components'] and m['components']['visual']['entry']['family']==family)
        metadata=copy.deepcopy(source);row=metadata['components']['visual']['entry']
        if fault=='word_bytes':row['whole_word_UTF8_bytes']=0
        elif fault=='photo_negative':row['complete_original_negative_response_EOS_targets']=1
        else:row['complete_original_negative_response_EOS_targets']=0
    with pytest.raises(ValueError):checkpoint_entry(metadata,source_identity_sha256='a'*64)
