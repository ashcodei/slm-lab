from pathlib import Path

import pytest

from slm_lab.admitted_public_prefix_episode_v1 import AdmittedPublicPrefixEpisode
from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus


@pytest.fixture(scope='module')
def corpus():return ExpandedExecutionCorpus(Path(__file__).resolve().parents[1])


def test_real_qualified_prefix_keeps_original_full_anchor_and_stopping_control(corpus):
    root=Path(__file__).resolve().parents[1]
    position=corpus.by_id['apps:apps:publisher-train:0001:solution:19:verified-generation-v1']
    episode=AdmittedPublicPrefixEpisode(corpus,position,3,root=root);prepared=episode.rollout_view()
    ids,labels=corpus.sequence(position);batch=episode.batch('cpu')
    assert batch['input_ids'][0].tolist()==ids and batch['labels'][0].tolist()==labels
    assert prepared['native_rollout_prefix_ids']+prepared['original_missing_suffix_and_EOS_ids']==ids
    assert len(prepared['native_rollout_prefix_ids'])>corpus.row(position)['prefix_positions']
    assert episode.reward([2])['actual_execution_reward']['reward']<1.
    actual=episode.reward(prepared['original_missing_suffix_and_EOS_ids'])
    assert actual['actual_execution_reward']['reward']==1. and actual['genuine_sampled_suffix_EOS']
    assert actual['complete_candidate_response']==corpus.row(position)['original']['response']
    assert not episode.binding()['all3721_source_prefixes_qualified']
    changed=episode.rollout_view();changed['native_rollout_prefix_ids'].append(2)
    assert episode.rollout_view()['native_rollout_prefix_ids']!=changed['native_rollout_prefix_ids']
    with pytest.raises(ValueError):episode.reward([7])


def test_unqualified_sources_or_levels_and_changed_episode_identity_are_rejected(corpus):
    root=Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError):AdmittedPublicPrefixEpisode(corpus,0,0,root=root)
    # This original source is outside the predetermined16-source controls.
    with pytest.raises(ValueError,match='not independently qualified'):AdmittedPublicPrefixEpisode(corpus,1,3,root=root)
    position=corpus.by_id['apps:apps:publisher-train:0001:solution:19:verified-generation-v1']
    episode=AdmittedPublicPrefixEpisode(corpus,position,3,root=root);episode.quarters=2
    with pytest.raises(ValueError):episode.rollout_view()
