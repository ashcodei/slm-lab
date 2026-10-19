"""One independently qualified public train prefix, with an unchanged CE anchor.

Only the16 already replayed original train sources and their2/3-quarter views
are qualified here. This is not admission of every3721-source curriculum view.
The original prompt, complete reference targets and published cases stay fixed.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .corpus import file_hash
from .expanded_execution_corpus_v1 import ExpandedExecutionCorpus
from .public_reference_prefix_curriculum_v1 import view


class AdmittedPublicPrefixEpisode:
    def __init__(self,corpus,position,quarters,*,root):
        if type(corpus) is not ExpandedExecutionCorpus or type(position) is not int or quarters not in [2,3] or type(quarters) is not int:
            raise ValueError('Exact admitted original train corpus, source and qualified prefix level required')
        self.corpus=corpus;self.position=position;self.quarters=quarters;root=Path(root).resolve()
        producer=root/'reports/public-reference-prefix-curriculum-v1'
        reader=root/'reports/public-reference-prefix-curriculum-v1-independent-reader'
        completed=json.loads((reader/'complete.json').read_text())
        if (completed.get('all14884_native_views_rederived_without_producer_view_function') is not True
                or completed.get('all16_preselected_original_references_freshly_reexecuted_and_passed') is not True
                or completed.get('all32_prefix_only_controls_do_not_earn_full_credit') is not True):
            raise ValueError('Actual complete source reconstruction and independent prefix/reference runtime controls required')
        for name,digest in completed['source_files_sha256'].items():
            if file_hash(Path(name))!=digest:raise ValueError('Original public-prefix qualification evidence changed')
        row=corpus.row(position)
        controls=[r for r in map(json.loads,(producer/'execution-controls.jsonl').open())
            if r['id']==row['id'] and r['retained_reference_quarters']==quarters]
        fresh=[r for r in map(json.loads,(reader/'fresh-execution-replay.jsonl').open())
            if r['id']==row['id'] and r['retained_reference_quarters']==quarters]
        if (len(controls)!=1 or len(fresh)!=1 or not controls[0]['proposed_conditional_episode_nontrivial']
                or not fresh[0]['independently_nontrivial']
                or fresh[0]['fresh_full_reference_execution']['reward']!=1.
                or fresh[0]['fresh_counterfactual_prefix_then_EOS_execution']['reward']>=1.):
            raise ValueError('This exact source/prefix is not independently qualified for a conditional training episode')
        ids,labels=corpus.sequence(position);prepared=view(corpus.tokenizer,row['original'],ids,labels,quarters)
        if (prepared['public_reference_prefix_text']!=controls[0]['supplied_public_reference_prefix']
                or not prepared['public_reference_prefix_ids']
                or any(t in [0,1,2] for t in ids[row['prefix_positions']:-1])):
            raise ValueError('A nonempty exact original public code prefix without embedded boundary tokens is required')
        self._view=prepared
        self._files={str(p):file_hash(p) for p in [reader/'complete.json',reader/'fresh-execution-replay.jsonl',
            producer/'plan.json',producer/'execution-controls.jsonl',Path(__file__)]}
        self._binding=dict(format='independently-qualified-public-train-prefix-episode-v1',original_position=position,
            original_source_id=row['id'],retained_reference_quarters=quarters,
            original_native_input_sha256=prepared['original_native_input_sha256'],
            full_original_reference_EOS_targets=row['response_EOS_targets'],
            supplied_original_reference_tokens=prepared['retained_reference_tokens'],
            native_rollout_prefix_ids_sha256=hashlib.sha256(np.asarray(prepared['native_rollout_prefix_ids'],dtype='<u2').tobytes()).hexdigest(),
            native_full_action_budget=prepared['complete_missing_suffix_training_budget'],
            full_original_reference_passed_and_prefix_only_full_credit_rejected=True,
            qualified_original_source_population=16,all3721_source_prefixes_qualified=False,
            qualification_source_files_sha256=dict(self._files))
        self.check()

    @property
    def tokenizer(self):return self.corpus.tokenizer

    def check(self):
        row=self.corpus.row(self.position);ids,labels=self.corpus.sequence(self.position)
        digest=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()
        if (self.position!=self._binding['original_position'] or self.quarters!=self._binding['retained_reference_quarters']
                or row['id']!=self._binding['original_source_id'] or digest!=self._binding['original_native_input_sha256']
                or ids!=self._view['complete_original_anchor_ids'] or labels!=self._view['complete_original_anchor_labels']
                or self._view['retained_reference_tokens']!=self._binding['supplied_original_reference_tokens']
                or self._view['native_rollout_prefix_ids']!=ids[:row['prefix_positions']+self._binding['supplied_original_reference_tokens']]
                or self._view['public_reference_prefix_ids']!=ids[row['prefix_positions']:row['prefix_positions']+self._binding['supplied_original_reference_tokens']]
                or self._view['complete_missing_suffix_training_budget']!=self._binding['native_full_action_budget']
                or self._view['native_rollout_prefix_ids']+self._view['original_missing_suffix_and_EOS_ids']!=ids):
            raise ValueError('Exact original source, qualified prefix and full CE anchor changed')
        return True

    def binding(self):return copy.deepcopy(self._binding)

    def rollout_view(self):
        self.check();return copy.deepcopy(self._view)

    def batch(self,device):
        self.check();return self.corpus.batch(self.position,device)

    def reward(self,actions,*,delivered_action_budget=None):
        self.check();budget=self._view['complete_missing_suffix_training_budget']
        if delivered_action_budget is not None:
            if type(delivered_action_budget) is not int or not 1<=delivered_action_budget<=budget:
                raise ValueError('Explicit actual rollout budget required')
            budget=delivered_action_budget
        if (not isinstance(actions,list) or not 1<=len(actions)<=budget
                or any(type(t) is not int or not 0<=t<32768 for t in actions)
                or 2 in actions[:-1] or actions[-1]!=2 and len(actions)!=budget):
            raise ValueError('Actual complete own-vocabulary suffix delivery requires generated EOS or its full declared budget')
        response=self.tokenizer.decode(self._view['public_reference_prefix_ids']+actions)
        actual=self.corpus.score(self.position,response,stopped_on_eos=actions[-1]==2)
        return dict(actual_execution_reward=actual,complete_candidate_response=response,
            actual_generated_suffix=self.tokenizer.decode(actions),
            public_reference_prefix_ids=list(self._view['public_reference_prefix_ids']),
            actual_generated_suffix_ids=list(actions),genuine_sampled_suffix_EOS=actions[-1]==2,
            supplied_prefix_is_original_public_training_reference=True,
            unassisted_capability_score=False,human_preferences=False,genuine_human_RLHF=False)
