"""Non-neural admission after exact public-data and own-policy delivery owners."""
import copy
import hashlib
import json
from pathlib import Path

FORMAT = 'own-whole-public-reasoned-code-stopping-owned-recipe-v1'
OPTIONS = dict(learning_rate=3e-5,gradient_clip=1.,maximum_microbatch_examples=2,
    maximum_padded_tokens=4096,gradient_bucket_values=1024**2,epochs=1,width=16,seed=990231)
OPTIMIZER = dict(betas=[.9,.95],eps=1e-8,weight_decay=.01)
STORAGE = dict(required_scratch_free_bytes=96*1024**3,required_home_free_bytes=60*1024**3,
    maximum_state_bytes=1536*1024**2,maximum_package_bytes=5*1024**3,
    maximum_assessment_bytes_per_endpoint=4*1024**3,maximum_journal_raw_bytes=1024**3,
    minimum_home_free_bytes=48*1024**3,minimum_scratch_free_bytes=40*1024**3)


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def live(owner):
    if set(owner) not in ({'pid','birth'},{'pid','ticks'}) or type(owner['pid']) is not int:
        raise ValueError('An exact process PID and original birth token are required')
    birth = str(owner.get('birth',owner.get('ticks')))
    try: fields = (Path('/proc')/str(owner['pid'])/'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError: return False
    return fields[19] == birth and fields[0] not in ('Z','X')


def prepare(root, *, data_plan, delivery, expected_data_owner, expected_delivery_owner, runtime_sources):
    root,data_plan,delivery = [Path(path).resolve() for path in (root,data_plan,delivery)]
    if (read(data_plan/'owner-plan.json')['owner'] != expected_data_owner
            or read(delivery/'waiter-plan.json')['owner'] != expected_delivery_owner):
        raise ValueError('Both exact original data-plan and standalone-delivery owners must remain bound')
    active = []
    for folder,owner in [(data_plan,expected_data_owner),(delivery,expected_delivery_owner)]:
        active.append(live(owner))
        if not active[-1] and ((folder/'failure.json').exists() or not (folder/'complete.json').is_file()):
            raise ValueError('An exact predecessor ended without complete successful original evidence; no fallback or automatic retry')
    if any(active): return None
    data = read(data_plan/'complete.json')
    delivered = read(delivery/'complete.json')
    if (data.get('owner') != expected_data_owner or data.get('complete') is not True
            or any(data.get(k) is not True for k in ['all_original_train_code_reasoning_and_EOS_targets_preserved',
                'original_tail_not_rounded','common_complete_source_schedule_for_both_profiles','reserved_optimization_forbidden',
                'public_synthetic_data_not_human_RL_feedback'])
            or data.get('production_training_admitted') is not False
            or delivered.get('owner') != expected_delivery_owner or delivered.get('complete') is not True
            or delivered.get('actual_registered_parameters') != 999706442
            or any(delivered.get(k) is not True for k in ['deployed_policy_and_original_trained_shared_reward_preserved',
                'old_checkpoints_or_training_features_not_needed_for_inference','original_four_L40PPO_and_full_assessment_artifacts_unchanged'])):
        raise ValueError('Complete native train proof and full original unfused policy/reward delivery required')
    capacity_path = data_plan/'complete-source-capacity-plan.json'
    capacity = read(capacity_path)
    budget = data['complete_original_budget']
    if (capacity.get('format') != 'own-public-reasoned-code-complete-capacity-source-selection-v1'
            or capacity.get('complete_source_selection') is not True or capacity.get('mechanics_only') is not False
            or capacity.get('source_selection_only') is not True or capacity.get('full_size_capacity_passed') is not False
            or capacity.get('all_four_replicas_receive_each_selected_source_in_declared_plan') is not True
            or capacity.get('complete_training_budget') != budget or capacity.get('epochs') != 1
            or capacity.get('width') != 16 or capacity.get('seed') != 990231
            or capacity.get('complete_training_schedule_identity') != data['complete_schedule_identity']
            or capacity.get('original_views') != ['code_only','reasoning']):
        raise ValueError('The entire actual admitted original train schedule and all source-extreme capacity cases required')
    for profile,factor in [('plain_tokens',1),('eos_weight4',4)]:
        expected = dict(targets=budget['response_EOS_targets'],EOS=budget['source_presentations'],
            weighted_targets=budget['response_EOS_targets']+(factor-1)*budget['source_presentations'])
        if data['actual_complete_profiles'][profile] != expected:
            raise ValueError('Both complete-response objective budgets must independently reconstruct')
    publication = Path(delivered['published_folder']).resolve()
    manifest = read(publication/'manifest.json')
    exporter = read(delivery/'export-complete.json')
    original = read(delivery/'plan.json')
    if (publication != root/'checkpoints/native-policy-reward-standalone-v1'
            or manifest.get('format') != 'own-standalone-policy-and-preserved-human-reward-delivery-v1'
            or manifest.get('actual_registered_parameters') != 999706442
            or manifest.get('trained_reward_feature_basis_preserved') is not True
            or manifest.get('old_checkpoint_or_features_needed_for_inference') is not False
            or set(manifest['files_sha256']) != {'model.pt','tokenizer.json'}
            or exporter.get('complete') is not True or exporter.get('actual_registered_parameters') != 999706442
            or exporter.get('all_policy_reward_and_auxiliary_values_and_original_full_outputs_CPUcold_exact') is not True
            or exporter.get('separate_actual_durable_CLI_completed') is not True
            or exporter.get('package_files_sha256') != manifest['files_sha256']
            or original.get('expected_completed_updates') != 10724):
        raise ValueError('Only the exact full completed self-contained own PPO policy and unchanged shared human reward may initialize')
    files = dict(runtime_sources)
    for incoming in [data['source_files_sha256'],data['output_files_sha256'],original['source_files_sha256']]:
        for path,digest in incoming.items():
            if path in files and files[path] != digest: raise ValueError('All original and new closed source identities must agree')
            files[path] = digest
    for folder,names in [(data_plan,['owner-plan.json','complete.json']),
            (delivery,['waiter-plan.json','plan.json','complete.json','export-complete.json']),
            (publication,['manifest.json'])]:
        for name in names: files[str(folder/name)] = sha(folder/name)
    for name,digest in manifest['files_sha256'].items(): files[str(publication/name)] = digest
    for path,digest in files.items():
        if sha(path) != digest: raise ValueError('Every exact completed source/result/package byte must remain unchanged')
    initializer = dict(checkpoint=str(publication/'model.pt'),checkpoint_sha256=manifest['files_sha256']['model.pt'],
        tokenizer=str(publication/'tokenizer.json'),tokenizer_sha256=manifest['files_sha256']['tokenizer.json'],
        sources=copy.deepcopy(manifest['sources']),completed_updates=10724)
    if initializer['sources']['tokenizer_sha256'] != initializer['tokenizer_sha256']:
        raise ValueError('The complete own-policy native tokenizer must remain exact')
    previous = original['PPO_plan']
    return dict(format=FORMAT,source_files_sha256=files,initializer=initializer,
        complete_capacity_plan_path=str(capacity_path),complete_capacity_plan_sha256=sha(capacity_path),
        complete_training_budget=budget,complete_training_schedule_identity=data['complete_schedule_identity'],
        public_code_manifest_sha256=capacity['dataset_manifest_sha256'],public_code_admission_sha256=capacity['native_admission_sha256'],
        exact_data_plan_owner=expected_data_owner,exact_delivery_owner=expected_delivery_owner,
        stage_options=copy.deepcopy(OPTIONS),optimizer=copy.deepcopy(OPTIMIZER),profiles=['plain_tokens','eos_weight4'],checkpoint_every=64,
        cold_inputs=copy.deepcopy(previous['cold_inputs']),cached_own_reference=copy.deepcopy(previous['cached_own_reference']),
        pulse_sources=copy.deepcopy(previous['pulse_sources']),storage_budget=copy.deepcopy(STORAGE),
        original_policy_adapters_critic_reward_and_multimodal_values_must_survive=True,
        own_output_human_alignment_or_model_quality_proved=False,actual_GPU_capacity_training_and_full_assessments_pending=True,
        original_all_scope_pipeline_preserved=True,automatic_retry=False,goal_achieved=False)
