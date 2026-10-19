import pytest
from slm_lab.vqa_pair_evaluation import human_agreement, paired_accuracy, select, validate_pairs


@pytest.mark.parametrize('votes,expected',[(0,0),(1,.3),(2,.6),(3,.9),(4,1),(10,1)])
def test_ten_human_leave_one_out_scoring_does_not_round_three_votes_to_full_credit(votes,expected):
    answers=[{'answer_id':i+1,'answer':'red' if i<votes else 'blue'} for i in range(10)]
    assert human_agreement('red',answers)==pytest.approx(expected)


def rows_and_pairs():
    rows=[];pairs=[]
    for i in range(3):
        ids=[]
        for side,answer in enumerate(['yes','no']):
            name=f'{i}:{side}';ids.append(name)
            rows.append({'id':name,'group_id':f'flickr:{2*i+side}','response':answer,'prompt':'same question?\nAnswer briefly.',
                         'original_human_annotation':{'answer_type':'yes/no','answers':[{'answer_id':j+1,'answer':answer} for j in range(10)]}})
        pairs.append({'id':str(i),'question_ids':ids,'canonical_question':'same question?'})
    return rows,pairs


def test_pair_score_cannot_reward_a_language_prior_that_answers_yes_for_both_images():
    assert paired_accuracy([{'id':'a','pair_id':'one','proper_exact':True},{'id':'b','pair_id':'one','proper_exact':False}])['both_answers_correct']==0


@pytest.mark.parametrize('change',['prompt','target','image','missing'])
def test_paired_assessment_rejects_text_hints_equal_targets_same_images_and_missing_rows(change):
    rows,pairs=rows_and_pairs()
    if change=='prompt':rows[1]['prompt']='different question?\nAnswer briefly.'
    elif change=='target':rows[1]['response']='yes'
    elif change=='image':rows[1]['group_id']=rows[0]['group_id']
    else:rows.pop(1)
    with pytest.raises(ValueError):validate_pairs(pairs,rows)


def test_pair_selection_counts_each_original_photo_once():
    rows,pairs=rows_and_pairs();rows[2]['group_id']=rows[0]['group_id']
    chosen,by_id=select(pairs,rows,3)
    groups=[by_id[key]['group_id'] for pair in chosen for key in pair['question_ids']]
    assert len(groups)==len(set(groups)) and len(chosen)==2
