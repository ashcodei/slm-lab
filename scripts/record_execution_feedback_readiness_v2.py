"""Record actual admitted feedback, completed checks and live four-L40 training."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess

from assess_joint_java_comparison_v1 import identity
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'data/execution-code-feedback-v2'
    admission=json.loads((source/'admission.json').read_text())
    if (not admission['passed'] or admission['complete_passing_originals']!=109
            or admission['finite_assertion_pairs']!=210 or admission['complete_original_registered_cases']!=1559):
        raise ValueError('Actual complete independently admitted feedback required')
    FileBindings(admission['source_files_sha256']).check(full=True)
    logs=[('mutation_and_raw_execution_labels','execution-code-feedback-v1-focused-tests.log',9),
        ('complete_reference_DPO','execution-preference-v1-focused-tests.log',4),
        ('all_actual_admitted_views','execution-feedback-reader-v1-focused-tests.log',5)]
    for _,name,number in logs:
        if str(number)+' passed' not in (ROOT/'reports'/name).read_text():raise ValueError('Completed focused checks required')
    latest=None
    for line in reversed((ROOT/'reports/curriculum-disk-cleanup-resumption-v1/joint.log').read_text().splitlines()):
        try:item=json.loads(line)
        except json.JSONDecodeError:continue
        if isinstance(item,dict) and 'step' in item and 'global_target_tokens' in item:
            latest=item;break
    actors=[]
    for pid in [2754743,2785192,2785198,2785199,2785200,2785201,130149,130151,2755925,2756113]:
        actor=identity(pid)
        if actor:
            proc=Path('/proc')/str(pid)
            actor['argv']=(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            actor['ppid']=int((proc/'stat').read_text().split(') ',1)[1].split()[1])
            actors.append(actor)
    # The torchrun launcher also names slm_lab.training. Count only its actual
    # direct worker children, then independently require their four GPU PIDs.
    workers=[a for a in actors if a['ppid']==2785192 and 'slm_lab.training' in a['argv']]
    if len(workers)!=4 or any(a['state'] in ['T','t','Z','X'] for a in workers):
        raise ValueError('Four actual live original training workers required')
    raw=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
    if set(map(int,raw.split()))!={a['pid'] for a in workers}:raise ValueError('Actual four CUDA process identities must match workers')
    status=json.loads((ROOT/'reports/dialogue-retention-v1/status.json').read_text())
    if status['steps']!=936 or status['status']!='training original library code with prior broad-code rehearsal':
        raise ValueError('Original complete936-update broad-code stage required')
    GPU=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
    paths=[source/'manifest.json',source/'admission.json',source/'pairs.jsonl',source/'passing-originals.jsonl',Path(__file__),
        ROOT/'src/slm_lab/execution_preference_v1.py',ROOT/'src/slm_lab/execution_feedback_reader_v1.py',
        ROOT/'tests/test_execution_code_feedback_v1.py',ROOT/'tests/test_execution_preference_v1.py',ROOT/'tests/test_execution_feedback_reader_v1.py']
    paths.extend(ROOT/'reports'/name for _,name,_ in logs)
    report=dict(observed_at=datetime.now(timezone.utc).isoformat(),format='own-execution-feedback-and-four-GPU-continuation-readiness-v2',
        goal_achieved=False,classification='progress',at_impasse=False,
        original_public_Java_training_modules=109,all109_original_reference_modules_pass=True,
        complete_original_publisher_cases=1559,mutation_attempts=sum(json.loads((source/'manifest.json').read_text())['mutation_qualifications'].values()),
        machine_verified_assertion_pairs=210,source_groups_with_pairs=84,
        all_raw_case_inventories_sources_mutations_BPE_masks_reconstructed=True,
        all_complete_programs_and_EOS_preserved_under4096=True,source_dataset_independently_admitted=True,
        new_feedback_used_in_model_updates=False,new_DPO_kernel_GPU_updates=0,human_feedback_or_RLHF_labels=False,
        ordinary_full_code_and_replay_populations_preserved=True,held_prompts_scoring_and_tests_unchanged=True,
        reference_DPO_gradient=dict(independent_FP64_direct_autograd_checks_passed=True,
            complete_response_sum_and_unequal_lengths=True,reference_policy_question_and_held_role_changes_reject_before_gradients=True,
            full706M_or857M_new_preference_backward_unexecuted=True,
            world4_auxiliary_training_checkpoint_source_binding_and_capacity_pending=True),
        implementation_checks=dict(total=18,**{name:number for name,_,number in logs}),
        source_files_sha256={str(p.resolve()):file_hash(p) for p in paths},
        original_continuation=dict(status=status,latest_logged_update=latest,
            nominal_target_exposures_since_stage_start=latest['step']*262080,actual_actors=actors,
            actual_four_GPU_workers=len(workers),GPU_observation=GPU),
        prior_main_controller_terminal=identity(2754742) is None,
        final_photo_manifest_exists=(ROOT/'data/vqa-real-pairs-v1/manifest.json').exists(),
        final30_source_bundle_exists=(ROOT/'data/codenet-final-source-contract-v4/manifest.json').exists(),
        verified_live_photo_source_pipeline=all(identity(p) is not None for p in [130149,2755925,2756113]),
        persistent_free_GiB=shutil.disk_usage(ROOT).free/1024**3,
        scope='Supplemental public Java finite-execution feedback. Mutants are machine-generated single-token changes, not new original public programs. Reference tests and implementation checks are not model capability scores. New feedback has no full-model updates or integration yet.',
        unresolved_goal_requirements=['Useful coding/all-round/high-end capability remains unverified; current primary functional code0/24.',
            'Bind actual full-model execution-preference training, full checkpoints and unchanged code/knowledge/image evaluations.',
            'Complete original broad-code/systems/rehearsal/photo/dialogue and fresh genuine human-feedback continuation.',
            'Finalize paired-photo30-family sources, exact ownbase/source correspondence, fresh baselines, actual four-L40 capacity/full-code production.',
            'Broader multilingual execution feedback, automatic routing, useful multimodal/RLHF behavior and independent final evaluation remain.'],
        preserved_failures=['reports/execution-code-feedback-v1-preparation-initial.log','reports/execution-code-feedback-v1-focused-tests-initial.log'],
        recording_preflight_note='The first inline status recorder counted the torchrun launcher as a fifth worker and stopped before writing. Current recorder checks direct worker parentage and actual CUDA PIDs.')
    atomic_json(ROOT/'reports/execution-code-feedback-v2-readiness.json',report)
    audit_path=ROOT/'reports/completion-audit.json';audit=json.loads(audit_path.read_text())
    audit.update(observed_at=report['observed_at'],goal_achieved=False,at_impasse=False,
        prior_goal_turn_classification=dict(classification='progress',reason='Both actual857M128-update experiments and full fixed code/knowledge assessments completed; neither improved coding; complete states/scores reparsed. Two actual source zero-output bugs found with newline/runtime qualifications.'),
        current_turn_classification=dict(classification='progress',reason='Original936-update broad/library code continuation is live on allfourL40s. All109public Java reference modules pass1559published cases;210single-token assertion-failing variants across84tasks independently admitted. Complete-program reference-DPO gradients match independent autograd;18focused checks pass. New feedback has no full-model update or promotion yet.'),
        execution_grounded_coding_feedback_current=str(ROOT/'reports/execution-code-feedback-v2-readiness.json'),
        latest_main_training_observation=report['original_continuation'],
        next_required_work='Continue original four-GPU broader-code/systems/source-finalized multimodal/dialogue and genuine-human-feedback pipeline. Integrate admitted execution feedback and complete-program DPO into a properly bound real-model experiment with full checkpoint/source/capacity and unchanged capability evidence; expand multilingual feedback. Preserve all30 original required sources and all final ownbase/coverage/baseline/production gates. High-end goal remains unachieved.')
    audit['current_requirement_assessment']['complete_required_sources_and_production_training']['evidence']='Original936-update broad/library-code continuation is live on allfourL40s. Original later systems/photo/dialogue/fresh human-feedback stages remain. New supplemental Java109positive modules/1559cases/210assertion pairs are independently admitted; no full-model preference update or production-contract integration yet. Original112609complete-code/final30-family/base/coverage/baseline/capacity/production requirements remain.'
    atomic_json(audit_path,audit)
    print(json.dumps(dict(originals=109,reference_cases=1559,assertion_pairs=210,focused_checks=18,
        actual_GPU_workers=len(workers),latest_broad_update=latest['step'],planned_updates=936,goal_achieved=False),indent=2))


if __name__=='__main__':main()
