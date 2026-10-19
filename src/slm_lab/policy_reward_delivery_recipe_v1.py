"""Source-only delivery admission after the exact complete four-L40 PPO run."""
import hashlib
import json
from pathlib import Path

from .exact_waiting_process_retirement_v1 import terminal

RESERVE=40*1024**3
STORAGE=dict(required_scratch_free_bytes=48*1024**3,required_home_free_bytes=45*1024**3,
    maximum_checkpoint_bytes=5*1024**3,maximum_publication_bytes=5*1024**3,
    maximum_metadata_bytes=64*1024**2)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare(root,*,PPO,expected_owner,runtime_sources):
    root,PPO=Path(root).resolve(),Path(PPO).resolve();waiter=read(PPO/'waiter-plan.json')
    if waiter['owner']!=expected_owner:raise ValueError('The exact original complete PPO source owner must remain bound')
    if not terminal(expected_owner):return None
    required=['plan.json','complete.json','restored.json','children.json','train/native-export.json',
        'matched-comparisons.json','matched-reasoning.json']
    if (PPO/'failure.json').exists() or any(not (PPO/n).is_file() for n in required):
        raise ValueError('The exact PPO owner ended without all original whole-source training/export/assessment evidence')
    plan,done,restored,native=[read(PPO/n) for n in ['plan.json','complete.json','restored.json','train/native-export.json']]
    if (plan['owner']!=expected_owner or done.get('complete') is not True
            or any(done.get(k) is not True for k in [
                'all10724four_L40_updates_and85786whole_original_prompts_complete',
                'every_original512action_capacity_private_replay_and_full_native_CPU_GPUcold_complete',
                'both_full_original_code_broad_QA_AI2D_MBPP_human_words_and_reserved_reasoning_assessments_complete'])
            or done.get('actual_native_values')!=984399690 or done.get('actual_research_policy_reward_values')!=999706442
            or plan.get('original_train_prompts')!=85786 or plan.get('global_options',{}).get('steps')!=10724
            or plan.get('global_options',{}).get('max_new_tokens')!=512 or plan.get('world_size')!=4
            or restored.get('all_registered_point_actors_terminal') is not True or any(not terminal(r) for r in read(PPO/'children.json'))
            or native.get('complete') is not True or native.get('completed_updates')!=10724
            or native.get('full85786prompt512action_epoch_complete') is not True):
        raise ValueError('All four actual original source epochs, private replay, native colds and full comparisons must be complete')
    source_proof=native.get('full_source_reconstruction',{})
    if (source_proof.get('every_actual_private_state_and_original_trajectory_checked') is not True
            or source_proof.get('actual_world')!=4 or source_proof.get('expected_completed')!=10724
            or len(source_proof.get('actual_complete_prefixes',[]))!=4
            or any(r.get('complete_updates')!=10724 or r.get('original_prompt_draws')!=85786
                for r in source_proof['actual_complete_prefixes'])):
        raise ValueError('The actual full original private source reconstruction proof is required')
    files=dict(runtime_sources);files.update(plan['source_files_sha256']);artifacts=[]
    for rank in range(4):
        report=PPO/'train'/f'rank-{rank}.json';row=read(report);cold_path=report.with_name(f'cold-{rank}.json');cold=read(cold_path)
        if (row.get('complete') is not True or row.get('rank')!=rank or row.get('world_size')!=4
                or row.get('actual_updates')!=10724 or row.get('actual_original_prompt_draws')!=85786
                or row.get('full85786prompt512action_epoch_complete') is not True
                or cold.get('complete') is not True or cold.get('physical_GPU')!=rank
                or cold.get('all_full4096text448photo140word_native_outputs_exact') is not True
                or cold.get('checkpoint_sha256')!=native['checkpoint_sha256']):
            raise ValueError('Every actual completed original private rank and whole original native GPUcold must remain')
        artifact=row['artifact'];path=Path(artifact['path']).resolve()
        if sha(path)!=artifact['sha256']:raise ValueError('An actual complete original private checkpoint changed')
        files[str(path)]=artifact['sha256'];files[str(report)]=sha(report);files[str(cold_path)]=sha(cold_path);artifacts.append(artifact)
    if len({r['common_sha256'] for r in artifacts})!=1 or len({r['private_RNG_sha256'] for r in artifacts})!=4:
        raise ValueError('Actual common actor/critic/Adam values and four distinct private generators must remain')
    paired=read(PPO/'matched-comparisons.json');reasoning=read(PPO/'matched-reasoning.json')
    if (paired.get('complete') is not True or paired.get('full_original_populations_and_all_image_conditions_preserved') is not True
            or reasoning.get('complete') is not True or reasoning.get('original_reserved_questions')!=289
            or not paired.get('source_files_sha256')):
        raise ValueError('Every complete original before/after comparison and all reserved reasoning results must remain')
    files.update(paired['source_files_sha256'])
    for endpoint in ['initial','PPO']:
        for name,count in [('full-QA.json',16961),('full-AI2D.json',3088),('full-original-MBPP.json',600),
                ('held-preferences.json',4642),('all-held-words.json',3429)]:
            path=PPO/endpoint/'assessment/standalone'/name
            if len(read(path).get('results',[]))!=count:raise ValueError('Every original complete source/condition assessment must remain')
            files[str(path)]=sha(path)
        path=PPO/endpoint/'assessment/all-reserved-reasoning.json'
        if len(read(path).get('results',[]))!=289:raise ValueError('Every original reserved reasoning source must remain at both endpoints')
        files[str(path)]=sha(path)
    publication=Path(done['durable_publication']).resolve();manifest=read(publication/'manifest.json')
    if (publication!=root/'checkpoints/native-shared-human-PPO-owned-v2'
            or manifest.get('format')!='own-four-L40-full-human-PPO-native-and-full-results-v2'
            or manifest.get('all85786original_prompts_once') is not True):
        raise ValueError('The original complete PPO native/result publication must remain intact')
    for name,digest in manifest['files_sha256'].items():files[str(publication/name)]=digest
    for name in ['waiter-plan.json',*required]:files[str(PPO/name)]=sha(PPO/name)
    files[str(publication/'manifest.json')]=sha(publication/'manifest.json')
    for path,digest in files.items():
        if sha(path)!=digest:raise ValueError('Every exact complete predecessor, source, result and new delivery byte must remain')
    return dict(format='own-complete-PPO-preserved-policy-reward-CPU-delivery-recipe-v1',
        source_files_sha256=files,original_PPO_reports=str(PPO),PPO_plan=plan,actual_original_private_artifacts=artifacts,
        expected_completed_updates=10724,expected_native_values=984399690,expected_policy_reward_values=999706442,
        original_PPO_owner=expected_owner,original_folded_native_and_every_assessment_unchanged=True,
        storage_budget=dict(STORAGE),expected_deployed_unique_values=999706442,
        trained_frozen_reward_basis_must_remain_exact=True,no_new_training_or_GPU_lease=True,
        model_quality_or_fresh_own_output_human_alignment_proved=False,goal_achieved=False)
