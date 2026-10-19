"""Non-neural full-source PPO admission after the exact learned-head owner."""
import ast
import copy
import json
from pathlib import Path

from .exact_waiting_process_retirement_v1 import terminal
from .native_reward_owned_recipe_v1 import completed_joint, sha, OPTIONS as HEAD_OPTIONS

FORMAT='own-four-L40-shared-native-full-human-PPO-owned-recipe-v2'
RESERVE=40*1024**3
BUDGET=dict(required_scratch_free_bytes=152*1024**3,required_home_free_bytes=48*1024**3,
    maximum_private_state_bytes=512*1024**2,maximum_native_bytes=5*1024**3,
    maximum_each_complete_assessment_bytes=1024**3,maximum_published_bytes=8*1024**3,
    maximum_metadata_bytes=64*1024**2)
HISTORY=dict(maximum_records=10724,maximum_compact_record_bytes=2*1024**2,
    maximum_expanded_record_bytes=2*1024**2,maximum_logical_bytes=64*1024**3,
    maximum_physical_bytes=16*1024**3,minimum_free_bytes=RESERVE)
SUITES=('code','broad','preferences','QA','AI2D','MBPP','words')


def read(path):return json.loads(Path(path).read_text())


def local_options(root):
    tree=ast.parse((Path(root)/'src/slm_lab/native_shared_ppo_v1.py').read_text())
    call=next(n.value for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='OPTIONS' for t in n.targets))
    if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!='dict' or call.args:
        raise ValueError('Exact fixed full original PPO options must be literal and readable without neural imports')
    return {k.arg:ast.literal_eval(k.value) for k in call.keywords}


def prepare(root,*,reward,expected_owner,joint,source_recipe,runtime_sources):
    root,reward,joint,source_recipe=map(lambda p:Path(p).resolve(),[root,reward,joint,source_recipe])
    waiter=read(reward/'waiter-plan.json')
    if waiter['owner']!=expected_owner:raise ValueError('The exact original full reward source owner must remain bound')
    if not terminal(expected_owner):return None
    required=['complete.json','restored.json','plan.json','children.json','fit/complete.json','cold.json']
    if (reward/'failure.json').exists() or any(not (reward/n).is_file() for n in required):
        raise ValueError('The exact reward owner ended without its entire original head epoch, cold and publication')
    plan,done,restored,fitted,cold=[read(reward/n) for n in ['plan.json','complete.json','restored.json','fit/complete.json','cold.json']]
    if (plan['owner']!=expected_owner or done.get('complete') is not True
            or any(done.get(k) is not True for k in ['full85786pair_reward_head_epoch_and_all4642initial_final_held_scores_complete',
                'all_four_original_L40_feature_capacities_and_complete_original_responses_checked',
                'full_standalone_native_export_and_head_score_cold_complete'])
            or done.get('actual_native_values')!=984399690 or done.get('actual_shared_policy_critic_and_reward_values')!=999706442
            or restored.get('all_registered_point_actors_terminal') is not True or any(not terminal(r) for r in read(reward/'children.json'))
            or fitted.get('complete') is not True or fitted.get('all85786original_human_pairs_once') is not True
            or fitted.get('actual_updates')!=2681 or fitted.get('all4642initial_and_final_held_scores_complete') is not True
            or plan.get('options')!=HEAD_OPTIONS or cold.get('complete') is not True
            or any(cold.get(k) is not True for k in ['all_original4642cached_feature_head_scores_replayed',
                'all_native_parameters_cold_load_exact','actual_complete_original_text_and_shared_reward_outputs_exact'])):
        raise ValueError('Every complete actual original reward capacity/source/head/held/cold scope must remain')
    joint_source=completed_joint(joint,waiter['exact_original_joint_owner'])
    if joint_source is None:raise ValueError('The whole preceding original joint experiment must also be terminal and complete')
    publication=Path(done['durable_head_publication']).resolve();manifest=read(publication/'manifest.json')
    if (publication!=root/'checkpoints/native-human-reward-owned-v1'
            or manifest.get('format')!='own-complete-current-native-human-reward-publication-v1'
            or manifest.get('sources')!=fitted['sources'] or manifest.get('options')!=HEAD_OPTIONS
            or manifest.get('initializer')!=joint_source['initializer']
            or set(manifest.get('files_sha256',{}))!={'head-bundle.pt','initial-held.json','final-held.json','cold.json'}
            or manifest['files_sha256']['head-bundle.pt']!=fitted['bundle_sha256']):
        raise ValueError('The exact whole durable trained human head, original native basis and complete held reports are required')
    files=dict(runtime_sources);files.update(joint_source['source_files_sha256'])
    files[str(reward/'waiter-plan.json')]=sha(reward/'waiter-plan.json')
    for name in required:files[str(reward/name)]=sha(reward/name)
    for name,digest in manifest['files_sha256'].items():
        path=publication/name
        if path.is_symlink() or sha(path)!=digest:raise ValueError('Every original durable head/publication byte must remain exact')
        files[str(path)]=digest
    files[str(publication/'manifest.json')]=sha(publication/'manifest.json')
    for name in ['initial-held.json','final-held.json']:
        if len(read(publication/name).get('results',[]))!=4642:raise ValueError('All4642held head scores must remain, without selecting a new rate')
    original_joint=read(joint/'plan.json')
    for key in ['cold_inputs','cached_own_reference','pulse_sources']:
        if not original_joint.get(key):raise ValueError('Every original complete native cold and full assessment source must remain')
    extras={original_joint['cold_inputs']['path']:original_joint['cold_inputs']['sha256'],
        original_joint['cached_own_reference']['report']:original_joint['cached_own_reference']['report_sha256']}
    extras.update({p:sha(p) for p in original_joint['pulse_sources'].values()})
    files.update(extras)
    source=read(source_recipe)
    if (source.get('complete') is not True or source.get('full_original_train_pairs')!=85786
            or source.get('original_held_pairs_kept_separate')!=4642 or source.get('maximum_prompt_plus512actions_and_scoring_EOS')!=1877
            or source.get('held_pairs_used_for_capacity_or_gradients')!=0 or source.get('unrepeated_source_epoch') is not True):
        raise ValueError('The actual complete original85786prompt/512action source recipe must remain')
    files.update(source['source_files_sha256']);files[str(source_recipe)]=sha(source_recipe)
    data=root/'data/whole-human-alignment-v2'
    if sha(data/'manifest.json')!=plan['human_manifest_sha256'] or sha(data/'admission.json')!=plan['human_admission_sha256']:
        raise ValueError('The complete original admitted human data must match the learned reward head')
    options=local_options(root)
    if source['options']!=options or options['steps']!=42893 or options['max_new_tokens']!=512 or options['prompts_per_update']!=2:
        raise ValueError('Preserve every original whole prompt and the full512action horizon')
    for p,h in files.items():
        if sha(p)!=h:raise ValueError('Every exact completed source, original result and new PPO runtime byte must remain')
    if sha(cold['checkpoint'])!=cold['checkpoint_sha256']:raise ValueError('The exact complete trained-head native baseline must remain')
    files[cold['checkpoint']]=cold['checkpoint_sha256']
    token_path=Path(cold['checkpoint']).parent/'tokenizer.json'
    if sha(token_path)!=cold['tokenizer_sha256']:raise ValueError('The baseline must retain its exact whole native tokenizer')
    files[str(token_path)]=cold['tokenizer_sha256']
    global_options=copy.deepcopy(options);global_options['steps']=10724
    return dict(format=FORMAT,source_files_sha256=files,initializer=joint_source['initializer'],
        baseline=dict(checkpoint=cold['checkpoint'],checkpoint_sha256=cold['checkpoint_sha256'],sources=cold['sources'],
            tokenizer=str(Path(cold['checkpoint']).parent/'tokenizer.json'),tokenizer_sha256=cold['tokenizer_sha256']),
        reward_bundle=dict(path=str(publication/'head-bundle.pt'),sha256=fitted['bundle_sha256'],sources=fitted['sources'],options=dict(HEAD_OPTIONS)),
        data=str(data),local_options=options,global_options=global_options,world_size=4,
        original_train_prompts=85786,held_human_pairs=4642,global_prompts_per_update=8,
        actor_settings=dict(rank=16,seed=997331,parameter_limit=10**9),private_seed=999271,
        history_budget=dict(HISTORY),storage_budget=dict(BUDGET),checkpoint_every=128,
        cold_inputs=copy.deepcopy(original_joint['cold_inputs']),cached_own_reference=copy.deepcopy(original_joint['cached_own_reference']),
        pulse_sources=copy.deepcopy(original_joint['pulse_sources']),suites=list(SUITES),tokenizer_sha256=cold['tokenizer_sha256'],
        expected_student_values=984399690,expected_policy_reward_values=999706442,
        exact_original_reward_owner=expected_owner,all_original_predecessors_and_assessment_populations_preserved=True,
        production_GPU_admitted=False,useful_own_output_human_alignment_proved=False,goal_achieved=False)
