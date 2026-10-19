"""Source/target and independent arithmetic readback of completed GPU scores.

No model forward, own-answer human judgment or fresh program execution is
performed. Preserve the distinction between reference-relative held margins,
raw response likelihood ordering and generated-code/knowledge performance.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/phase-human-preference-v8'
OUT=ROOT/'reports/human-v8-held-source-arithmetic-readback-v1.json'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def credit(value):return 1. if value>0 else .5 if value==0 else 0.


def arithmetic(rows,beta):
    margins=[(r['policy_chosen_logp']-r['policy_rejected_logp'])-(r['reference_chosen_logp']-r['reference_rejected_logp']) for r in rows]
    x=[-beta*m for m in margins]
    return dict(pairs=len(rows),pair_mean_DPO=math.fsum(max(t,0)+math.log1p(math.exp(-abs(t))) for t in x)/len(rows),
        implicit_preference_accuracy=math.fsum(map(credit,margins))/len(rows),implicit_preference_ties=sum(m==0 for m in margins),
        raw_sequence_preference_accuracy=math.fsum(credit(r['policy_chosen_logp']-r['policy_rejected_logp']) for r in rows)/len(rows),
        token_mean_preference_accuracy=math.fsum(credit(r['policy_chosen_logp']/r['chosen_response_EOS_targets']-r['policy_rejected_logp']/r['rejected_response_EOS_targets']) for r in rows)/len(rows),
        shorter_response_prior_accuracy=math.fsum(credit(r['rejected_response_EOS_targets']-r['chosen_response_EOS_targets']) for r in rows)/len(rows),
        chosen_token_mean_NLL=-math.fsum(r['policy_chosen_logp'] for r in rows)/sum(r['chosen_response_EOS_targets'] for r in rows),
        rejected_token_mean_NLL=-math.fsum(r['policy_rejected_logp'] for r in rows)/sum(r['rejected_response_EOS_targets'] for r in rows),
        complete_context_tokens=sum(r['complete_context_tokens'] for r in rows),
        complete_response_EOS_targets=sum(r['chosen_response_EOS_targets']+r['rejected_response_EOS_targets'] for r in rows))


def same(actual,wanted):
    if set(actual)!=set(wanted):raise ValueError('Every original human score/count must remain')
    for k,v in wanted.items():
        if type(v) is int:
            if actual[k]!=v:raise ValueError('Original human count differs: '+k)
        elif not math.isclose(actual[k],v,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Original human score arithmetic differs: '+k)


def main():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19:raise ValueError('A fresh CUDA-hidden idle readback is required')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0))
    import torch
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.dense_human_preference_evaluation_v1 import information
    torch.set_num_threads(1)
    complete=json.loads((BASE/'complete.json').read_text())
    if complete.get('all_held_human_pairs_and_native_development_assessments_complete') is not True:raise ValueError('All three original full human assessments must finish first')
    held=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','valid','evaluation')
    if len(held)!=4642:raise ValueError('Every one of the original4642held public human pairs is required')
    originals=[information(held,p) for p in range(len(held))];files={str(BASE/'complete.json'):sha(BASE/'complete.json')};profiles={};rows_by_profile={};reference=None
    for name in ['initial','chosen_CE','human_DPO']:
        folder=BASE/name;path=folder/'held-preferences.json';report=json.loads(path.read_text());files[str(path)]=sha(path)
        rows=report['results'];beta=report['DPO_beta']
        if [r['original_held_position'] for r in rows]!=list(range(4642)) or beta!=.1:raise ValueError('Every original full held pair and beta must remain')
        for original,row in zip(originals,rows):
            if any(row[k]!=v for k,v in original.items()):raise ValueError('Original source/prompt/whole response/EOS/fingerprint differs')
            if any(not math.isfinite(row[p+'_'+f+'_logp']) or row[p+'_'+f+'_logp']>0 for p in ['policy','reference'] for f in ['chosen','rejected']):
                raise ValueError('All original complete response likelihoods must remain finite and nonpositive')
            margin=(row['policy_chosen_logp']-row['policy_rejected_logp'])-(row['reference_chosen_logp']-row['reference_rejected_logp']);x=-beta*margin
            expected=dict(implicit_preference_margin=margin,pair_DPO=max(x,0)+math.log1p(math.exp(-abs(x))),implicit_preference_credit=credit(margin),
                raw_sequence_preference_credit=credit(row['policy_chosen_logp']-row['policy_rejected_logp']),
                token_mean_preference_credit=credit(row['policy_chosen_logp']/row['chosen_response_EOS_targets']-row['policy_rejected_logp']/row['rejected_response_EOS_targets']),
                shorter_response_prior_credit=credit(row['rejected_response_EOS_targets']-row['chosen_response_EOS_targets']))
            if any(row[k]!=v for k,v in expected.items()):raise ValueError('Every original per-pair preference and length-control value must reconstruct')
        references=[(r['reference_chosen_logp'],r['reference_rejected_logp']) for r in rows]
        if reference is None:reference=references
        elif reference!=references:raise ValueError('The identical own frozen reference must score each original response in every arm')
        same(report['overall'],arithmetic(rows,beta))
        for source in {r['source_blob'] for r in rows}:same(report['by_source'][source],arithmetic([r for r in rows if r['source_blob']==source],beta))
        for rank in range(4):
            shard_path=folder/('held-preferences-shard-'+str(rank)+'.json');shard=json.loads(shard_path.read_text())
            if (shard['rank']!=rank or shard['world']!=4 or shard['original_held_positions']!=list(range(rank,4642,4))
                    or shard['results']!=rows[rank::4] or shard['held_gradients'] is not False):raise ValueError('Every original held shard and raw complete score row must remain')
            journal=Path(shard['raw_journal']);files[str(shard_path)]=sha(shard_path);files[str(journal)]=sha(journal)
            if files[str(journal)]!=shard['raw_journal_sha256'] or [json.loads(line) for line in journal.open()]!=shard['results']:
                raise ValueError('Every original raw journal byte and complete row must match')
        codes={}
        for frame in ['completion','instruction']:
            p=folder/('code-'+frame+'.json');d=json.loads(p.read_text());files[str(p)]=sha(p)
            codes[frame]={k:d[k] for k in ['task_count','syntax_pass_rate','functional_pass_at_1','efficiency_stress_pass_rate','independent_code_reexecution_still_required']}
        profiles[name]=dict(held=arithmetic(rows,beta),native_development_code_producer_observations=codes);rows_by_profile[name]=rows
    paired=Counter()
    for a,b in zip(rows_by_profile['chosen_CE'],rows_by_profile['human_DPO']):
        paired['SFT_correct' if a['implicit_preference_credit']==1 else 'SFT_other']+=1
        paired['DPO_correct' if b['implicit_preference_credit']==1 else 'DPO_other']+=1
        if a['implicit_preference_credit']!=b['implicit_preference_credit']:
            paired['DPO_only_correct' if b['implicit_preference_credit']>a['implicit_preference_credit'] else 'SFT_only_correct']+=1
    for suffix in ['manifest.json','preferences.valid.jsonl','preferences.valid.tokens.u16']:
        p=held.root/suffix;files[str(p)]=sha(p)
    files[str(Path(__file__).resolve())]=sha(__file__)
    if torch.cuda.is_initialized():raise ValueError('An arithmetic/source readback must not acquire a GPU context')
    result=dict(complete=True,at=datetime.now(timezone.utc).isoformat(),actual_original_held_pairs_per_profile=4642,actual_profiles=profiles,
        actual_paired_reference_relative_discordance=dict(paired),all_original_source_prompt_response_EOS_and_target_fingerprints_reconstructed=True,
        all_original_raw_score_rows_journals_and_aggregates_reconstructed=True,all_three_actual_reference_logp_tables_bit_equal=True,
        DPO_vs_chosen_SFT_reference_relative_accuracy_percentage_point_difference=100*(profiles['human_DPO']['held']['implicit_preference_accuracy']-profiles['chosen_CE']['held']['implicit_preference_accuracy']),
        reference_relative_improvement_is_not_raw_response_preference_or_own_generation_quality=True,
        actual_neural_likelihoods_not_independently_replayed=True,independent_generated_code_reexecution_still_required=True,
        fresh_own_output_human_judgments=False,models_constructed=0,neural_forwards=0,optimizer_updates=0,GPU_contexts=0,
        maintenance_metadata_only_not_training_storage_admission=True,source_and_evidence_files_sha256=files,candidate_promoted=False,goal_achieved=False)
    OUT.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(complete=True,pairs_per_profile=4642,DPO_reference_relative_accuracy=profiles['human_DPO']['held']['implicit_preference_accuracy'],
        SFT_reference_relative_accuracy=profiles['chosen_CE']['held']['implicit_preference_accuracy'],paired=dict(paired),goal_achieved=False)))


if __name__=='__main__':main()
