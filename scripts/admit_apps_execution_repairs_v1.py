"""Admit independently reconstructed full human-target repair CE data only."""
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/APPS-execution-repairs-v1'
REPORTS=ROOT/'reports/APPS-execution-repairs-v1'


def main():
    target=DATA/'admission.json'
    if target.exists():raise ValueError('Preserve original repair admission; no implicit overwrite')
    proof_path=REPORTS/'independent-candidate-readback.json';proof=json.loads(proof_path.read_text())
    for key in ['all_actual_repair_rows_tokens_and_response_EOS_masks_reconstructed',
            'all_targets_byte_equal_original_human_programs',
            'every_original_train_identity_and_all_case_result_independently_reconciled',
            'every_actual_single_token_edit_and_first_failed_value_independently_reconstructed']:
        if proof.get(key) is not True:raise ValueError('Complete independent actual repair source proof required: '+key)
    if (proof['exact_named_held_matches_or_normalization_reviews'] or proof['source_groups_crossing_internal_validation']
            or proof['original_target_or_case_bytes_clipped'] or proof['human_preferences']
            or proof['external_model_teacher_used'] or proof['model_generated_examples']
            or proof['candidate_manifest_sha256']!=file_hash(DATA/'manifest.json')):
        raise ValueError('Complete named-screened original human-target train-only candidate required')
    plan=json.loads((REPORTS/'plan.json').read_text());files=dict(plan['source_files_sha256'])
    for name,digest in proof['files_sha256'].items():
        if name in files and files[name]!=digest:raise ValueError('Conflicting source proof digests')
        files[name]=digest
    manifest=json.loads((DATA/'manifest.json').read_text())
    files.update({str(DATA/name):digest for name,digest in manifest['files_sha256'].items()})
    for p in [Path(__file__).resolve(),proof_path,DATA/'manifest.json',REPORTS/'plan.json']:
        files[str(p)]=file_hash(p)
    guard=FileBindings(files);guard.check(full=True)
    pack=CompleteCodePack(DATA,'train',purpose='audit')
    with gzip.open(DATA/'original-rows.jsonl.gz','rt') as stream:rows=list(map(json.loads,stream))
    if (len(rows)!=len(pack) or len(rows)!=proof['complete_fit_repair_examples']
            or [r['token_pack_index'] for r in rows]!=list(range(len(pack)))
            or any(r['split']!='train' or r['publisher_split']!='train' or r['human_preferences'] for r in rows)):
        raise ValueError('Every original train-only repair pack row and source position required')
    unique_ids={r['original_source_program_id'] for r in rows}
    unique_bodies={r['original_source_body_sha256'] for r in rows}
    maximum=max(r['complete_context_tokens'] for r in rows)
    admission=dict(format='own-complete-human-target-execution-repair-CE-admission-v1',
        admitted_at=datetime.now(timezone.utc).isoformat(),dataset_manifest_sha256=file_hash(DATA/'manifest.json'),
        admitted_to_training=True,admitted_objective='Supervised complete original-human-target code repair CE',
        complete_examples=len(rows),complete_source_problems=proof['unique_source_problems'],
        original_source_program_IDs=len(unique_ids),distinct_original_human_program_byte_sequences=len(unique_bodies),
        complete_response_EOS_targets=proof['complete_response_targets'],maximum_complete_context_tokens=maximum,
        original_train_groups_only=True,original_targets_byte_unchanged=True,
        every_mutation_and_actual_complete_case_counterexample_reconstructed=True,
        complete_original_prompt_program_witness_target_EOS_preserved=True,
        named_exact_held_source_matches=0,held_screen_scope=proof['held_screen_scope'],
        semantic_or_prior_public_corpus_independence_certified=False,
        finite_case_passing_not_general_correctness_or_efficiency_proof=True,
        human_preferences=False,genuine_RLHF_labels=False,machine_execution_feedback=True,
        external_model_teacher_used=False,model_generated_examples=0,
        preserved_overcontext_repairs_outside_training=proof['preserved_whole_context_reviews'],
        source_files_sha256=files,current_active_training_changed=False,
        model_updates=0,model_capability_gain=False,goal_achieved=False)
    guard.check(full=True);atomic_json(target,admission)
    loaded=CompleteCodePack(DATA,'train',purpose='training')
    checks=sorted({0,len(rows)-1,max(range(len(rows)),key=lambda i:rows[i]['complete_context_tokens']),
        max(range(len(rows)),key=lambda i:rows[i]['complete_response_targets'])})
    for i in checks:
        ids,labels=loaded.sequence(i);start=next(j for j,x in enumerate(labels) if x!=-100)
        if (len(ids)!=rows[i]['complete_context_tokens'] or ids[-1]!=2
                or labels[:start]!=[-100]*start or labels[start:]!=ids[start:]
                or len(ids)-start!=rows[i]['complete_response_targets']):
            raise ValueError('Actual admitted complete-code reader boundary masks differ')
    try:CompleteCodePack(DATA,'validation',purpose='training')
    except ValueError as error:
        if 'Held source roles cannot be used' not in str(error):raise
    else:raise ValueError('A held optimization role was accepted')
    atomic_json(REPORTS/'admitted-reader-readback.json',dict(observed_at=datetime.now(timezone.utc).isoformat(),
        admission_sha256=file_hash(target),source_manifest_sha256=file_hash(DATA/'manifest.json'),
        complete_examples=len(loaded),actual_boundary_positions=checks,held_optimization_role_rejected=True,
        original_source_program_IDs=len(unique_ids),distinct_original_human_program_byte_sequences=len(unique_bodies),
        all6520_original_source_positions_verified=True,maximum_complete_context_tokens=maximum,
        full_independent_all_row_BPE_mask_proof_sha256=file_hash(proof_path),
        training_started=False,model_updates=0,model_capability_gain=False,goal_achieved=False))
    print(json.dumps({k:v for k,v in admission.items() if k!='source_files_sha256'},indent=2),flush=True)


if __name__=='__main__':main()
