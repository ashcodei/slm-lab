"""Both complete native responses and truthful public-derivative provenance."""
import copy
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from slm_lab.function_semantic_stress_v1 import DONORS,cases,observe
from slm_lab.public_semantic_source_contrast_v1 import make
from test_function_semantic_stress_v1 import sources,donors

ROOT=Path(__file__).resolve().parents[1]


def test_actual_bad_public_sources_create_complete_native_positive_negative_masks_and_original_rights():
    tokenizer=Tokenizer.from_file(str(ROOT/'data/direct-algorithm-function-contracts-admitted-v1/tokenizer.json'));alternatives=donors()
    for row in sources():
        name=row['source_candidate']['function']
        if name not in DONORS:continue
        bad=observe(row['response'],name,cases(row));donor=alternatives[DONORS[name]]
        candidate,ids,labels,pair=make(row,donor,bad,tokenizer)
        prefix=pair['native_prompt_ids'];chosen=pair['chosen_complete_response_EOS_ids'];rejected=pair['rejected_complete_response_EOS_ids']
        assert ids==prefix+chosen and labels==[-100]*len(prefix)+chosen
        assert chosen[-1]==rejected[-1]==2 and max(len(ids),len(prefix)+len(rejected))<=4096
        assert tokenizer.decode(chosen[:-1],skip_special_tokens=False)==candidate['response']
        assert tokenizer.decode(rejected[:-1],skip_special_tokens=False)==row['response']
        assert candidate['response'].startswith(donor['response']) and candidate['rights']==row['rights']
        assert candidate['donor_rights']==donor['rights'] and candidate['dataset_admitted_to_training'] is False
        assert candidate['human_preferences'] is False and candidate['on_policy_RL'] is False


def test_missing_wrong_or_falsified_negative_and_validation_donor_reject_before_native_construction():
    tokenizer=Tokenizer.from_file(str(ROOT/'data/direct-algorithm-function-contracts-admitted-v1/tokenizer.json'))
    row=next(r for r in sources() if r['source_candidate']['function']=='perfect_cube');donor=donors()[DONORS['perfect_cube']]
    bad=observe(row['response'],'perfect_cube',cases(row))
    changed=copy.deepcopy(bad);changed['case_passes']=[True]*len(changed['case_passes'])
    with pytest.raises(ValueError):make(row,donor,changed,tokenizer)
    changed=copy.deepcopy(bad);changed['program_sha256']='0'*64
    with pytest.raises(ValueError):make(row,donor,changed,tokenizer)
    changed=copy.deepcopy(donor);changed['source_candidate']['role']='valid'
    with pytest.raises(ValueError):make(row,changed,bad,tokenizer)
