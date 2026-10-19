"""Admit a separate screened publisher-label CE view, never execution rewards."""
from collections import Counter
from datetime import datetime,timezone
import gzip
import json
import os
from pathlib import Path

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/code-contests-text-screened-CE-view-v1'
AUDIT=ROOT/'reports/CodeContests-native-held-text-v2'
READER=ROOT/'reports/CodeContests-held-text-view-v1-readback/complete.json'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only separately admitted CE view required')
    native_path=ROOT/'reports/code-contests-complete-curriculum-v1-readback/complete.json'
    native=json.loads(native_path.read_text());reader=json.loads(READER.read_text());audit=json.loads((AUDIT/'complete.json').read_text())
    view_path=AUDIT/'future-training-view-candidate.json';view=json.loads(view_path.read_text())
    if (not native['all_selected_actual_publisher_code_and_complete_question_bytes_and_BPE_tokens_and_EOS_masks_rebuilt']
            or not reader['all67007_actual_native_input_hashes_roles_positions_and_component_view_dispositions_rederived']
            or not reader['all_reported_positive_witnesses_rebuilt_without_Aho_automaton_or_inverted_candidate_search']
            or reader['actual_unflagged_native_budgets']!=view['actual_unflagged_native_budgets']
            or audit['remaining_original_train_presentations']!=len(view['original_training_pack_positions'])):
        raise ValueError('Every complete native target and held-source component disposition must reconcile')
    pack=CompleteCodePack(view['original_pack'],'train','audit')
    files={str(p):file_hash(p) for p in [native_path,READER,AUDIT/'complete.json',view_path,
        AUDIT/'additional-component-exclusions.json',pack.folder/'manifest.json',ROOT/'src/slm_lab/code_contests_CE_view_v1.py']}
    files.update(native['source_files_sha256']);files.update(reader['source_files_sha256'])
    files.update({str(pack.folder/n):v for n,v in pack.manifest['files_sha256'].items()})
    files.update({str(p):file_hash(p) for p in python_sources(ROOT,[Path(__file__),ROOT/'src/slm_lab/code_contests_CE_view_v1.py'])})
    guard=FileBindings(files);positions=view['original_training_pack_positions']
    if len(positions)!=58488 or len(set(positions))!=len(positions) or any(int(pack.index[p]['role'])!=0 for p in positions):
        raise ValueError('All58488unique screened original train positions required')
    selected=set(positions);languages=Counter();problems=set();groups=set()
    with gzip.open(pack.folder/'original-rows.jsonl.gz','rt') as rows:
        for position,line in enumerate(rows):
            if position not in selected:continue
            row=json.loads(line)
            if row['split']!='train' or row['publisher_split']!='train':raise ValueError('Held source cannot be admitted to CE training')
            languages[row['language']]+=1;problems.add(row['source_problem_id']);groups.add(row['group_id'])
    OUT.mkdir();at=datetime.now(timezone.utc).isoformat()
    atomic_json(OUT/'original-training-pack-positions.json',positions)
    manifest=dict(format='own-CodeContests-source-screened-complete-CE-view-v1',at=at,
        original_pack=str(pack.folder),original_pack_manifest_sha256=file_hash(pack.folder/'manifest.json'),
        positions_sha256=file_hash(OUT/'original-training-pack-positions.json'),
        native_counts=view['actual_unflagged_native_budgets']['train'],actual_original_problem_sources=len(problems),
        actual_original_source_components=len(groups),original_language_presentations=dict(languages),
        supervision='Unmodified original publisher-correct program text and native EOS, full prompt masked',
        source_text_audit=str(AUDIT/'complete.json'),source_text_and_view_readback=str(READER),
        all_known_source_quarantines_applied_before_optimization=True,
        publisher_correctness_labels_not_fresh_local_reference_execution=True,
        generic_literal_and_lexical_copy_exclusions_are_conservative_not_semantic_equivalence_labels=True,
        source_ancestry_and_semantic_independence_not_certified=True,
        original_held_populations_and_original_pack_unchanged=True,
        original_source_terms=str(ROOT/'data/code-contests-original-train-cache-v2/publisher-README.md'),
        code_license='Publisher Apache-2.0, retaining third-party terms caveats',
        non_code_license='Publisher CC-BY-4.0, retaining attribution and third-party terms caveats',
        new_model_training_started=False,execution_rewards_admitted=False,model_updates=0,goal_achieved=False)
    atomic_json(OUT/'manifest.json',manifest)
    guard.check(full=True)
    files.update({str(OUT/'original-training-pack-positions.json'):file_hash(OUT/'original-training-pack-positions.json'),
        str(OUT/'manifest.json'):file_hash(OUT/'manifest.json')})
    atomic_json(OUT/'admission.json',dict(at=at,admitted_to_CE_training=True,admitted_to_execution_reward_training=False,
        manifest_sha256=file_hash(OUT/'manifest.json'),source_files_sha256=files,
        all_complete_native_targets_and_source_exclusion_witnesses_read_back=True,
        CE_uses_original_publisher_correctness_labels_not_new_local_execution=True,
        execution_reward_and_human_preference_uses_require_separate_admission=True,
        no_original_pack_admission_granted=True,no_held_source_role_admitted=True,
        producer_negative_exposure_search_not_independently_repeated=True,
        new_model_training_started=False,model_updates=0,candidate_promoted=False,goal_achieved=False))
    print(json.dumps(dict(CE_view=str(OUT),native_train_presentations=len(positions),original_problem_sources=len(problems),
        complete_context_positions=manifest['native_counts']['context_positions'],admitted_to_CE_only=True,model_updates=0)),flush=True)


if __name__=='__main__':main()
