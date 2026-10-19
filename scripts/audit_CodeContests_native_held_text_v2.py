"""Audit every complete candidate native input against named held-source text.

Preserves every held assessment case. Emits additional component exclusions for
a future training view; neither modifies the pack nor admits it to training.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from tokenizers import Tokenizer

from slm_lab.ai2d_exposure_patterns_v1 import registry as ai2d_registry
from slm_lab.code_contests_curriculum_candidates_v1 import normalized_program_hash, question_hash
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_description_v1 import parse_description
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.code_tasks import TASKS
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.held_text_overlap_v1 import HeldTextOverlap, normalize
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.source_jsonl_records_v1 import records

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/code-contests-complete-curriculum-candidate-v1'
PRODUCER=ROOT/'reports/code-contests-complete-curriculum-candidate-v1'
OUT=ROOT/'reports/CodeContests-native-held-text-v2'
READBACK=ROOT/'reports/code-contests-complete-curriculum-v1-readback/complete.json'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('TOKENIZERS_PARALLELISM') != 'false' or OUT.exists():
        raise ValueError('Fresh CPU-hidden complete native-source audit required')
    readback=json.loads(READBACK.read_text())
    if (not readback['all_selected_actual_publisher_code_and_complete_question_bytes_and_BPE_tokens_and_EOS_masks_rebuilt']
            or readback['retained_native_presentations_rebuilt'] != 67007 or readback['dataset_admitted_to_training'] is not False):
        raise ValueError('All67007retained complete native targets must already be rebuilt from original source')
    pack=CompleteCodePack(DATA,'train','audit');tokenizer=Tokenizer.from_file(str(DATA/'tokenizer.json'))
    profile_path=PRODUCER/'all-original-problem-source-dispositions.json'
    profiles=json.loads(profile_path.read_text()); by_problem={r['id']:r for r in profiles}
    plan_path=PRODUCER/'plan.json';producer_plan=json.loads(plan_path.read_text())
    direct=[Path(__file__),READBACK,profile_path,plan_path,DATA/'manifest.json',
        ROOT/'data/full-held-question-pool-v2/question-registry.jsonl',
        ROOT/'data/ai2d-all-original-test-eval-v2/questions.jsonl',
        ROOT/'data/verified-code-source-cache-v1/sanitized-mbpp.json',
        ROOT/'data/verified-code-expanded-v1/reserved-mbpp.jsonl',
        ROOT/'data/public-apps-original-inventory-v1/train-problems.json',
        ROOT/'data/public-apps-original-inventory-v1/test-problems.json',
        ROOT/'data/public-apps-complete-pack-v2/original-rows.jsonl.gz',
        ROOT/'data/codenet-internal-splits-v1/problem-roles.json',
        ROOT/'reports/codenet-original-inspection-v1/inspection.json',
        ROOT/'tests/test_held_text_overlap_v1.py',ROOT/'reports/held-text-overlap-v1-tests.log',
        ROOT/'tests/test_source_jsonl_records_v1.py',ROOT/'reports/source-jsonl-records-v1-tests.log']
    direct += python_sources(ROOT,[Path(__file__),ROOT/'tests/test_held_text_overlap_v1.py'])
    files={str(p):file_hash(p) for p in direct}
    files.update({str(DATA/name):value for name,value in pack.manifest['files_sha256'].items()})
    registry=[]; protected_programs=defaultdict(list)
    for row in records(ROOT/'data/full-held-question-pool-v2/question-registry.jsonl'):
        registry.append(dict(audit_id=row['audit_id'],benchmark=row['benchmark'],question=row['question']))
    ai2d=list(records(ROOT/'data/ai2d-all-original-test-eval-v2/questions.jsonl'))
    registry += [{k:r[k] for k in ['audit_id','benchmark','question']} for r in ai2d_registry(ai2d)]
    validation=set()
    with gzip.open(ROOT/'data/public-apps-complete-pack-v2/original-rows.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line)
            if row['split']=='validation':validation.add(row['source_problem_id'])
    apps=ROOT/'data/public-apps-original-inventory-v1';missing_descriptions=[]
    for role in ['test','validation']:
        inventory=json.loads((apps/('test-problems.json' if role=='test' else 'train-problems.json')).read_text())
        for identity,item in inventory.items():
            if role=='validation' and identity not in validation:continue
            path=(ROOT/'data/public-apps-publisher-test-evaluation-v1/original-test' if role=='test' else apps/'original-train')/identity/'question.txt'
            expected=item['files']['question.txt']['sha256']
            if file_hash(path)!=expected:raise ValueError('Every original held APPS text byte must remain unchanged')
            files[str(path)]=expected
            registry.append(dict(audit_id='APPS:'+role+':'+identity,benchmark='APPS:'+role,question=path.read_bytes().decode()))
    roles=json.loads((ROOT/'data/codenet-internal-splits-v1/problem-roles.json').read_text())
    inspection=json.loads((ROOT/'reports/codenet-original-inspection-v1/inspection.json').read_text())
    cache=Path(inspection['metadata_cache']);member_path=cache/'retained-members.jsonl'
    if file_hash(member_path)!=inspection['retained_member_index_sha256']:raise ValueError('Original CodeNet source member index changed')
    files[str(member_path)]=inspection['retained_member_index_sha256']
    members={r['cache_relative_path']:r for r in records(member_path)}
    for identity,row in roles.items():
        if row['split']=='train':continue
        relative='problem_descriptions/'+identity+'.html'
        if relative not in members:
            missing_descriptions.append(dict(problem_id=identity,role=row['split'],reason='No original HTML member retained by publisher archive inventory'))
            continue
        path=cache/relative;expected=members[relative]['sha256']
        if file_hash(path)!=expected:raise ValueError('Original held CodeNet description changed')
        files[str(path)]=expected;text=parse_description(path.read_bytes().decode())['text']
        if not text:
            missing_descriptions.append(dict(problem_id=identity,role=row['split'],reason='Original retained HTML has no exported visible text'))
            continue
        registry.append(dict(audit_id='CodeNet:'+row['split']+':'+identity,benchmark='CodeNet:'+row['split'],question=text))
    for row in json.loads((ROOT/'data/verified-code-source-cache-v1/sanitized-mbpp.json').read_text()):
        if 11<=row['task_id']<=510:
            identity='sanitized-MBPP:'+str(row['task_id'])
            registry.append(dict(audit_id=identity,benchmark='sanitized-MBPP',question=row['prompt']))
            protected_programs[normalized_program_hash(row['code'])].append(identity)
    for row in records(ROOT/'data/verified-code-expanded-v1/reserved-mbpp.jsonl'):
        protected_programs[normalized_program_hash(row['code'])].append('reserved-MBPP:'+str(row['task_id']))
    for name,args,spec,scaling,python_code,js_code in TASKS:
        identity='local-code:'+name
        registry.append(dict(audit_id=identity,benchmark='local-code',question=spec))
        for program in [python_code,js_code]:protected_programs[normalized_program_hash(program)].append(identity)
    index=HeldTextOverlap(registry)
    guard=FileBindings(files);OUT.mkdir();started=time.monotonic()
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,
        source_files_sha256=files,actual_native_presentations=67007,actual_native_context_positions=96920929,
        actual_original_held_stems_by_family=dict(Counter(r['benchmark'] for r in registry)),
        exact_program_reference_fingerprints=len(protected_programs),lexical_copy_configuration=index.configuration,
        all_current_candidate_train_and_internal_validation_inputs_scanned=True,
        source_scope_not_all4432447_raw_publisher_positive_programs=True,
        known_held_registry_gold_outputs_and_model_scores_not_used=True,
        generic_literal_match_not_necessarily_same_problem=True,
        no_match_not_semantic_or_complete_historical_independence=True,
        no_held_assessment_case_removed=True,dataset_admitted_to_training=False,models_constructed=0,model_updates=0))
    atomic_json(OUT/'unchanged-held-text-registry.json',registry)
    atomic_json(OUT/'original-CodeNet-held-description-dispositions.json',dict(held_problem_IDs=sum(r['split']!='train' for r in roles.values()),
        exported_nonempty_descriptions=sum(r['benchmark'].startswith('CodeNet:') for r in registry),missing_or_empty=missing_descriptions,
        original_ID_quarantine_already_preserved_without_text=True))
    by_id={r['audit_id']:r for r in registry};counts=Counter(); problem_flags={};literal_counts=Counter();near_counts=Counter();program_counts=Counter()
    ledger=OUT/'every-native-presentation-exposure.jsonl'
    try:
        with gzip.open(DATA/'original-rows.jsonl.gz','rt') as rows,ledger.open('x') as output:
            for position,line in enumerate(rows):
                row=json.loads(line);entry=pack.index[position];offset=int(entry['offset']);length=int(entry['length'])
                raw=pack.ids[offset:offset+length];digest=hashlib.sha256(raw.tobytes()).hexdigest()
                if (row['token_pack_index']!=position or row['complete_input_ids_sha256']!=digest
                        or row['publisher_split']!='train' or row['split'] not in {'train','validation'}):
                    raise ValueError('Every actual complete candidate input and sealed original role required')
                source_id=row['source_problem_id'];profile=by_problem[source_id]
                if row['split']!=profile['split'] or row['group_id']!=profile['group_id']:raise ValueError('Original source component assignment differs')
                description=row['prompt'].split('\n\n',1)[1]
                if question_hash(description)!=profile['complete_original_question_normalized_sha256']:
                    raise ValueError('Whole source question must reconstruct exactly for lexical-copy check')
                if source_id not in problem_flags:
                    near=index.lexical_copies(description)
                    problem_flags[source_id]=dict(source_problem_id=source_id,group_id=row['group_id'],split=row['split'],
                        original_question_sha256=hashlib.sha256(description.encode()).hexdigest(),
                        complete_question_lexical_copy_candidates=near,literal_held_stems={},normalized_reference_program_matches={})
                    near_counts.update(by_id[k]['benchmark'] for k in near)
                flag=problem_flags[source_id]
                text=tokenizer.decode(raw[1:-1].astype('int64').tolist(),skip_special_tokens=False)
                literal=index.literal(text)
                for key,value in literal.items():
                    flag['literal_held_stems'].setdefault(key,dict(original_presentation_id=row['id'],token_pack_index=position,**value))
                    literal_counts[by_id[key]['benchmark']]+=1
                program_matches=protected_programs.get(normalized_program_hash(row['response']),[])
                for key in program_matches:
                    flag['normalized_reference_program_matches'].setdefault(key,dict(original_presentation_id=row['id'],token_pack_index=position))
                    program_counts[key.split(':',1)[0]]+=1
                result=dict(id=row['id'],source_problem_id=source_id,group_id=row['group_id'],split=row['split'],token_pack_index=position,
                    complete_native_context_positions=length,complete_native_input_ids_sha256=digest,
                    decoded_normalized_actual_native_text_sha256=hashlib.sha256(normalize(text).encode()).hexdigest(),
                    literal_held_stem_matches=literal,complete_question_lexical_copy_candidate_ids=list(flag['complete_question_lexical_copy_candidates']),
                    normalized_held_reference_program_matches=program_matches)
                output.write(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
                counts['native_presentations']+=1;counts['native_context_positions']+=length
                counts['presentations_with_literal_stems']+=bool(literal)
                counts['presentations_with_lexical_question_copy_candidates']+=bool(flag['complete_question_lexical_copy_candidates'])
                counts['presentations_with_normalized_reference_program_matches']+=bool(program_matches)
                if (position+1)%1024==0:
                    atomic_json(OUT/'progress.json',dict(owner=owner,actual_counts=dict(counts),source_problems=len(problem_flags),
                        elapsed_seconds=time.monotonic()-started,models_constructed=0,model_updates=0))
                    if (position+1)%8192==0:print(json.dumps(dict(presentations=position+1,source_problems=len(problem_flags),counts=dict(counts))),flush=True)
        if counts['native_presentations']!=67007 or counts['native_context_positions']!=96920929:
            raise ValueError('All original candidate inputs and context positions must be scanned')
        blocked=defaultdict(set)
        for flag in problem_flags.values():
            if flag['literal_held_stems']:blocked[flag['group_id']].add('literal_held_stem_in_actual_native_input')
            if flag['complete_question_lexical_copy_candidates']:blocked[flag['group_id']].add('declared_lexical_question_copy_candidate')
            if flag['normalized_reference_program_matches']:blocked[flag['group_id']].add('normalized_held_reference_program_match')
        eligible=[];exclusions=[];view_counts=defaultdict(Counter)
        with gzip.open(DATA/'original-rows.jsonl.gz','rt') as rows:
            for position,line in enumerate(rows):
                row=json.loads(line);reason=sorted(blocked.get(row['group_id'],set()))
                if reason:
                    exclusions.append(dict(id=row['id'],token_pack_index=position,source_problem_id=row['source_problem_id'],
                        original_split=row['split'],group_id=row['group_id'],reasons=reason))
                else:
                    for k,v in [('presentations',1),('context_positions',int(pack.index[position]['length'])),
                            ('response_EOS_targets',int(pack.index[position]['length'])-int(pack.index[position]['target_start']))]:view_counts[row['split']][k]+=v
                    if row['split']=='train':eligible.append(position)
        guard.check(full=True)
        atomic_json(OUT/'all-source-problem-text-exposure-flags.json',list(problem_flags.values()))
        atomic_json(OUT/'additional-component-exclusions.json',dict(component_reasons={k:sorted(v) for k,v in blocked.items()},
            all_excluded_original_presentations=exclusions,no_original_source_bytes_or_held_case_modified=True))
        atomic_json(OUT/'future-training-view-candidate.json',dict(format='own-CodeContests-text-screened-view-candidate-v1',
            original_pack=str(DATA),original_pack_manifest_sha256=file_hash(DATA/'manifest.json'),
            original_training_pack_positions=eligible,complete_source_role_preserved=True,
            actual_unflagged_native_budgets={k:dict(v) for k,v in view_counts.items()},
            actual_blocked_components=len(blocked),additional_excluded_presentations=len(exclusions),
            exposure_signal_exclusions_conservative_not_semantic_equivalence_labels=True,
            unchanged_held_assessment_population=True,independent_witness_and_view_readback_pending=True,
            publisher_correctness_label_not_local_execution=True,CE_and_execution_reward_admission_not_granted=True,
            dataset_admitted_to_training=False,model_updates=0,goal_achieved=False))
        receipts=[OUT/'plan.json',OUT/'unchanged-held-text-registry.json',OUT/'original-CodeNet-held-description-dispositions.json',
            ledger,OUT/'all-source-problem-text-exposure-flags.json',OUT/'additional-component-exclusions.json',OUT/'future-training-view-candidate.json',READBACK]
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
            actual_counts=dict(counts),actual_original_source_problems=len(problem_flags),actual_named_held_stems=len(registry),
            literal_match_presentation_instances_by_held_family=dict(literal_counts),lexical_question_copy_instances_by_held_family=dict(near_counts),
            normalized_reference_program_match_instances_by_family=dict(program_counts),additional_blocked_components=len(blocked),
            additional_excluded_original_presentations=len(exclusions),actual_unflagged_native_budgets={k:dict(v) for k,v in view_counts.items()},
            original_train_candidate_presentations=65395,remaining_original_train_presentations=len(eligible),
            every_actual_candidate_input_scanned_without_clipping=True,all_original_source_and_held_assessment_cases_preserved=True,
            source_files_sha256={str(p):file_hash(p) for p in receipts},
            independent_witness_and_view_readback_pending=True,semantic_or_all_ancestry_independence_not_certified=True,
            dataset_admitted_to_training=False,models_constructed=0,model_updates=0,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),counts=dict(counts),additional_blocked_components=len(blocked),
            remaining_train_presentations=len(eligible),admitted_to_training=False)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),model_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
