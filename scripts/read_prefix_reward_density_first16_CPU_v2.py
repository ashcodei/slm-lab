"""Read a complete first16 episode block and independently apply the original runtime protocol to every candidate.

The ongoing neural diagnostic is not stopped or modified. A copied immutable
prefix binds the observed raw token deliveries; the original ledger may append.
These are public training-source conditional tasks, never a held coding score.
"""
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/prefix-reward-density-full-CPU-v1'
OUT=ROOT/'reports/prefix-reward-density-first16-CPU-v2-readback'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only complete first-block neural delivery readback required')
    import torch
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    torch.set_num_threads(1)
    plan_path=SOURCE/'plan.json';plan=json.loads(plan_path.read_text())
    if plan['planned_actual_neural_episodes']!=40 or plan['matched_generation_budget']!=512:
        raise ValueError('The exact fixed full-model diagnostic protocol is required')
    source_path=SOURCE/'actual-neural-episodes.jsonl';lines=source_path.read_bytes().splitlines(keepends=True)
    if len(lines)<16 or not all(line.endswith(b'\n') for line in lines[:16]):raise ValueError('A complete first16 episode block must exist')
    raw=b''.join(lines[:16]);records=[json.loads(line) for line in lines[:16]]
    if [r['task'] for r in records]!=plan['tasks'][:16]:raise ValueError('Every exact original first-block task and private seed required')
    corpus=ExpandedExecutionCorpus(ROOT);OUT.mkdir()
    captured=OUT/'actual-neural-first16.jsonl';captured.write_bytes(raw)
    replay=OUT/'fresh-program-executions.jsonl';summary=defaultdict(list)
    keys=['reward','status','all_original_published_examples_passed','all_augmented_reference_observations_passed',
        'full_reference_stream_passed','added_probe_passes','scored_observations','scored_passes',
        'all_finite_original_cases_passed','informative_cases_passed','informative_cases']
    def signature(score):return {k:score[k] for k in keys if k in score}
    with replay.open('x') as output:
        for record in records:
            task=record['task'];actual=record['actual_rollout'];source=corpus.row(task['position'])
            ids,labels=corpus.sequence(task['position']);start=source['prefix_positions'];keep=record['public_reference_prefix_tokens']
            quarter=task['retained_reference_quarters'];actions=actual['actual_generated_action_ids']
            hint=ids[start:start+keep];prefix=ids[:start+keep]
            original_sha=hashlib.sha256(torch.tensor(ids,dtype=torch.int64).numpy().astype('<u2').tobytes()).hexdigest()
            if (source['id']!=task['source_id'] or record['original_source_native_input_sha256']!=original_sha
                    or quarter not in [0,3] or not 0<=keep<=len(ids[start:-1])*quarter//4
                    or keep and (len(ids)-start-keep-1<16 or not corpus.tokenizer.decode(hint).endswith('\n'))
                    or actual['supplied_public_reference_prefix_ids']!=hint or actual['original_native_prefix_ids']!=prefix
                    or actual['sampler']!=task['sampler'] or actual['seed']!=task['seed'] or actual['horizon']!=512
                    or not 1<=len(actions)<=512 or 2 in actions[:-1] or any(type(t) is not int or not 0<=t<32768 for t in actions)
                    or actual['stopped_on_generated_EOS']!=(actions[-1]==2)
                    or actions[-1]!=2 and len(actions)!=512
                    or actual['actual_generated_action_tokens']!=len(actions)
                    or actual['complete_candidate_response']!=corpus.tokenizer.decode(hint+actions)
                    or actual['actual_generated_suffix']!=corpus.tokenizer.decode(actions)):
                raise ValueError('Every real native source, public hint and sampled suffix/EOS delivery must reconstruct')
            cached=actual['actual_cached_logps'];full=actual['actual_full_forward_logps']
            if (len(cached)!=len(actions) or len(full)!=len(actions)
                    or any(not math.isfinite(x) or x>1e-7 for x in cached+full)
                    or not math.isclose(max(abs(a-b) for a,b in zip(cached,full)),actual['cached_full_max_absolute_difference'],abs_tol=1e-12)):
                raise ValueError('Complete actual cached/full probability observations must agree')
            fresh=corpus.score(task['position'],actual['complete_candidate_response'],stopped_on_eos=actual['stopped_on_generated_EOS'])
            if signature(fresh)!=signature(record['actual_execution_reward']):raise ValueError('Independent full original execution outcome differs')
            if fresh['reward']==1. and not actual['stopped_on_generated_EOS']:raise ValueError('Full training credit requires actual sampled EOS')
            key=str(quarter)+':'+task['sampler'];summary[key].append(dict(source_id=source['id'],reward=fresh['reward'],
                genuine_EOS=actual['stopped_on_generated_EOS'],actual_generated_action_tokens=len(actions)))
            output.write(json.dumps(dict(task=task,fresh_original_program_execution=fresh,
                actual_generated_suffix_and_public_prefix_reconstructed=True,
                original_runtime_protocol_reapplied_including_compile_scope_failures=True),ensure_ascii=False,separators=(',',':'))+'\n')
            output.flush();os.fsync(output.fileno())
    with source_path.open('rb') as source:
        if source.read(len(raw))!=raw:raise ValueError('Observed ongoing neural ledger prefix changed during readback')
    aggregates={}
    for key,values in summary.items():
        if len(values)!=4:raise ValueError('Every complete four-source condition and sampler block required')
        aggregates[key]=dict(episodes=4,positive_rewards=sum(v['reward']>0 for v in values),
            full_finite_case_credit_with_actual_EOS=sum(v['reward']==1. for v in values),
            genuine_EOS_deliveries=sum(v['genuine_EOS'] for v in values),mean_reward=math.fsum(v['reward'] for v in values)/4,
            actual_source_outcomes=values)
    files=[Path(__file__),plan_path,SOURCE/'model-loaded.json',captured,replay,ROOT/'src/slm_lab/native_prefix_rollout_CPU_v1.py',
        ROOT/'src/slm_lab/expanded_execution_corpus_v1.py']
    result=dict(at=datetime.now(timezone.utc).isoformat(),complete=True,original_predeclared_first_block_episodes=16,
        actual_own_neural_student_values=984399690,actual_own_checkpoint_completed_updates=2048,
        all_actual_public_prefix_and_neural_suffix_token_deliveries_reconstructed=True,
        all16_actual_candidate_programs_independently_rechecked_with_same_original_outcomes=True,
        summary=aggregates,source_ledger_prefix_observation=dict(path=str(source_path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
            original_ledger_may_append_but_captured_prefix_is_immutable=True),
        source_files_sha256={str(p):file_hash(p) for p in files},
        full40_episode_neural_diagnostic_still_in_progress=True,within_source_four_sample_reward_variation_not_yet_measured=True,
        no_new_neural_forward_or_weight_update_by_this_reader=True,model_updates=0,
        public_training_conditional_exercises_not_unassisted_held_coding_or_frontier_quality=True,
        no_statistically_established_curriculum_effect_or_GPU_RL_admission=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(OUT/'complete.json',result);print(json.dumps({k:{j:v for j,v in value.items() if j!='actual_source_outcomes'} for k,value in aggregates.items()}),flush=True)


if __name__=='__main__':main()
