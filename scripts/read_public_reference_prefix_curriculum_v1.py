"""Independently reconstruct all native curriculum views and replay train controls."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/public-reference-prefix-curriculum-v1-independent-reader'
PRODUCER=ROOT/'reports/public-reference-prefix-curriculum-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only native/control readback required')
    import numpy as np
    import torch
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    torch.set_num_threads(1)
    result=json.loads((PRODUCER/'complete.json').read_text());plan=json.loads((PRODUCER/'plan.json').read_text())
    for name,digest in result['source_files_sha256'].items():
        if file_hash(Path(name))!=digest:raise ValueError('Completed curriculum source evidence changed')
    corpus=ExpandedExecutionCorpus(ROOT)
    values=[json.loads(s) for s in (PRODUCER/'native-views.jsonl').open()]
    if len(values)!=14884:raise ValueError('All proposed quarter-level source views required')
    by_id=defaultdict(dict)
    for row in values:
        key=row['retained_reference_quarters']
        if key not in [3,2,1,0] or key in by_id[row['id']]:raise ValueError('Every source/level exactly once required')
        by_id[row['id']][key]=row
    totals={q:Counter() for q in [3,2,1,0]}
    if set(by_id)!=set(corpus.by_id):raise ValueError('The complete3721 original train source population is required')
    for position,source in enumerate(corpus.rows):
        original=source['original'];ids,labels=corpus.sequence(position)
        start=next(i for i,v in enumerate(labels) if v!=-100);target_count=len(ids)-start
        sha=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()
        if (original['split']!='train' or original['publisher_split']!='train' or
                corpus.tokenizer.decode(ids[start:-1],skip_special_tokens=False)!=original['response'] or
                labels[:start]!=[-100]*start or labels[start:]!=ids[start:]):
            raise ValueError('Whole unchanged original train reference and complete anchor required')
        retained=[]
        for quarter in [3,2,1,0]:
            row=by_id[source['id']][quarter];keep=row['retained_reference_tokens'];missing=target_count-keep
            hint=corpus.tokenizer.decode(ids[start:start+keep],skip_special_tokens=False)
            if (row['original_id']!=original['id'] or row['original_native_input_sha256']!=sha
                    or row['complete_context_positions']!=len(ids) or row['complete_original_anchor_targets']!=target_count
                    or row['original_native_prompt_positions']!=start or row['conditional_rollout_prefix_positions']!=start+keep
                    or row['missing_original_response_EOS_targets']!=missing or not 0<=keep<=len(ids[start:-1])*quarter//4
                    or keep and (missing-1<16 or not hint.endswith('\n') or not original['response'].startswith(hint))
                    or row['conditional_rollout_action_budget']<missing or row['conditional_rollout_prefix_positions']+row['conditional_rollout_action_budget']>4096
                    or row['admitted_to_RL'] is not False or row['training_only'] is not True):
                raise ValueError('Every exact original hinted prefix, full remaining suffix, anchor and resource bound must reconstruct')
            retained.append(keep);totals[quarter].update(views=1,complete_original_anchor_targets=target_count,
                complete_original_context_positions=len(ids),retained_reference_tokens=keep,
                missing_original_response_EOS_targets=missing,views_with_actual_hint=int(keep>0))
        if retained!=sorted(retained,reverse=True) or retained[-1]!=0:
            raise ValueError('The fixed curriculum must remove hints monotonically to the unchanged unassisted source')
    if {str(k):dict(v) for k,v in totals.items()}!=result['quarter_level_counts']:
        raise ValueError('All published native-view aggregates must reconstruct')
    controls=[json.loads(s) for s in (PRODUCER/'execution-controls.jsonl').open()]
    if len(controls)!=32 or {r['id'] for r in controls}!=set(plan['controls_selected_before_execution']):
        raise ValueError('All original source-preselected control episodes required')
    OUT.mkdir();ledger=OUT/'fresh-execution-replay.jsonl';replayed={};seen=set();passed=0
    score_keys=['reward','status','all_original_published_examples_passed','all_augmented_reference_observations_passed',
        'full_reference_stream_passed','added_probe_passes','scored_observations','scored_passes',
        'all_finite_original_cases_passed','informative_cases_passed','informative_cases']
    def signature(score):return {k:score[k] for k in score_keys if k in score}
    with ledger.open('x') as output:
        for old in controls:
            position=corpus.by_id[old['id']];source=corpus.row(position)
            if old['position']!=position:raise ValueError('Original control source cursor changed')
            key=(old['id'],old['retained_reference_quarters'])
            if key in seen:raise ValueError('Duplicate control episode')
            seen.add(key);row=by_id[old['id']][old['retained_reference_quarters']]
            ids,labels=corpus.sequence(position);start=row['original_native_prompt_positions'];keep=row['retained_reference_tokens']
            hint=corpus.tokenizer.decode(ids[start:start+keep],skip_special_tokens=False)
            if hint!=old['supplied_public_reference_prefix'] or not source['original']['response'].startswith(hint):
                raise ValueError('Actually executed prefix must equal exact original native source prefix')
            if old['id'] not in replayed:
                replayed[old['id']]=corpus.score(position,source['original']['response'],stopped_on_eos=True)
            reference=replayed[old['id']];prefix=corpus.score(position,hint,stopped_on_eos=True)
            if (signature(reference)!=signature(old['full_original_reference_execution'])
                    or signature(prefix)!=signature(old['counterfactual_prefix_then_EOS_execution'])):
                raise ValueError('Fresh independent train-control outcomes differ')
            eligible=reference['reward']==1. and prefix['reward']<1. and keep>0
            if eligible!=old['proposed_conditional_episode_nontrivial'] or old['admitted_to_RL'] is not False:
                raise ValueError('Every nontrivial episode verdict and unadmitted status must reconstruct')
            passed+=int(eligible)
            fresh=dict(id=old['id'],retained_reference_quarters=old['retained_reference_quarters'],
                fresh_full_reference_execution=reference,fresh_counterfactual_prefix_then_EOS_execution=prefix,
                independently_nontrivial=eligible,model_generations=0,model_updates=0)
            output.write(json.dumps(fresh,ensure_ascii=False,separators=(',',':'))+'\n');output.flush();os.fsync(output.fileno())
    if len(replayed)!=16 or passed!=32:raise ValueError('All actual balanced positive references and nontrivial hinted controls required')
    files=[Path(__file__),PRODUCER/'complete.json',PRODUCER/'plan.json',PRODUCER/'native-views.jsonl',
        PRODUCER/'execution-controls.jsonl',ledger,ROOT/'src/slm_lab/expanded_execution_corpus_v1.py']
    report=dict(at=datetime.now(timezone.utc).isoformat(),complete=True,all14884_native_views_rederived_without_producer_view_function=True,
        all3721_original_train_programs_and_complete_CE_anchors_preserved_at_all_four_levels=True,
        all16_preselected_original_references_freshly_reexecuted_and_passed=True,
        all32_counterfactual_prefix_then_EOS_controls_freshly_reexecuted=True,
        all32_prefix_only_controls_do_not_earn_full_credit=True,
        all32_actual_conditional_episode_nontrivial_verdicts_rederived=True,
        source_files_sha256={str(p):file_hash(p) for p in files},
        no_student_rollouts_or_reward_density_or_unassisted_capability_measurement=True,
        full_curriculum_execution_admission_actual_student_rollouts_GPU_capacity_and_training_still_required=True,
        current_live_and_queued_training_unchanged=True,model_generations=0,model_updates=0,
        admitted_to_RL=False,candidate_promoted=False,goal_achieved=False)
    atomic_json(OUT/'complete.json',report);print(json.dumps(dict(all_native_views_rederived=14884,
        actual_reference_programs_reexecuted=16,actual_prefix_controls_reexecuted=32,nontrivial_controls=passed,
        actual_student_generations=0,model_updates=0)),flush=True)


if __name__=='__main__':main()
