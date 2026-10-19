"""Rebuild all positive exposure witnesses and all native-view dispositions.

Direct literal searches and explicit shingle intersections do not use the
producer's automaton or inverted candidate search. Negative results remain the
producer's full scan, not a second independent exhaustive search.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re

from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.held_text_overlap_v1 import normalize
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.source_jsonl_records_v1 import records

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'reports/CodeContests-native-held-text-v2'
DATA=ROOT/'data/code-contests-complete-curriculum-candidate-v1'
OUT=ROOT/'reports/CodeContests-held-text-view-v1-readback'


def word_sets(text):
    words=re.findall(r'[^\W_]+(?:_[^\W_]+)*',normalize(text),re.UNICODE)
    return set(zip(words,words[1:],words[2:],words[3:],words[4:]))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only whole-view readback required')
    completion_path=AUDIT/'complete.json';complete=json.loads(completion_path.read_text());plan=json.loads((AUDIT/'plan.json').read_text())
    config=plan['lexical_copy_configuration']
    if (config['word_shingle_width']!=5 or config['maximum_held_document_frequency']!=25
            or config['minimum_distinct_informative_shared_shingles']!=24
            or config['minimum_shorter_informative_set_coverage']!=.7 or config['minimum_longer_informative_set_coverage']!=.3):
        raise ValueError('Unchanged declared literal and lexical-copy protocol required')
    guard=FileBindings({**complete['source_files_sha256'],str(completion_path):file_hash(completion_path),
        **{str(p):file_hash(p) for p in python_sources(ROOT,[Path(__file__)])}})
    pack=CompleteCodePack(DATA,'train','audit');tokenizer=Tokenizer.from_file(str(DATA/'tokenizer.json'))
    registry=json.loads((AUDIT/'unchanged-held-text-registry.json').read_text());held={r['audit_id']:r for r in registry}
    if (len(held)!=32953 or dict(Counter(r['benchmark'] for r in registry))!=plan['actual_original_held_stems_by_family']):
        raise ValueError('Every original held stem and source-family disposition required')
    raw_sets={identity:word_sets(row['question']) for identity,row in held.items()}
    frequencies=Counter(value for values in raw_sets.values() for value in values)
    ignored={value for value,count in frequencies.items() if count>25}
    informative={identity:values-ignored for identity,values in raw_sets.items()}
    if len(ignored)!=config['actual_high_frequency_shingles_ignored_only_for_lexical_copy_check']:
        raise ValueError('Original held-only document frequency exclusions differ')
    profiles={r['source_problem_id']:r for r in json.loads((AUDIT/'all-source-problem-text-exposure-flags.json').read_text())}
    OUT.mkdir();owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,
        source_files_sha256={str(p):file_hash(p) for p in [Path(__file__),completion_path,AUDIT/'plan.json',DATA/'manifest.json']},
        actual_all_original_native_presentations=67007,all_positive_witnesses_and_view_dispositions_required=True,
        negative_search_not_independently_repeated=True,model_updates=0))
    counts=Counter();near_by_problem={};literal_witnesses=[];near_witnesses=[];blocked=defaultdict(set)
    expected_positions=[];expected_exclusions=[];budgets=defaultdict(Counter);observed_problem_literal=defaultdict(dict)
    ledger=iter(records(AUDIT/'every-native-presentation-exposure.jsonl'))
    with gzip.open(DATA/'original-rows.jsonl.gz','rt') as rows:
        for position,line in enumerate(rows):
            row=json.loads(line);observed=next(ledger);entry=pack.index[position]
            offset=int(entry['offset']);length=int(entry['length']);actual=pack.ids[offset:offset+length]
            if (observed['token_pack_index']!=position or observed['id']!=row['id'] or observed['source_problem_id']!=row['source_problem_id']
                    or observed['group_id']!=row['group_id'] or observed['split']!=row['split']
                    or observed['complete_native_context_positions']!=length
                    or observed['complete_native_input_ids_sha256']!=hashlib.sha256(actual.tobytes()).hexdigest()):
                raise ValueError('Every native source position, source role and actual input byte must reconstruct')
            source=row['source_problem_id'];profile=profiles[source];group=row['group_id']
            if source not in near_by_problem:
                text=row['prompt'].split('\n\n',1)[1]
                if hashlib.sha256(text.encode()).hexdigest()!=profile['original_question_sha256']:raise ValueError('Original source question differs')
                query=word_sets(text)-ignored;witnesses={}
                for identity,reported in profile['complete_question_lexical_copy_candidates'].items():
                    other=informative[identity];shared=len(query&other)
                    actual_overlap=dict(distinct_shared_informative_shingles=shared,candidate_informative_shingles=len(query),
                        held_informative_shingles=len(other),shorter_set_coverage=shared/min(len(query),len(other)),
                        longer_set_coverage=shared/max(len(query),len(other)))
                    if actual_overlap!=reported or shared<24 or actual_overlap['shorter_set_coverage']<.7 or actual_overlap['longer_set_coverage']<.3:
                        raise ValueError('Every positive lexical-copy witness must reconstruct from actual unchanged questions')
                    witnesses[identity]=actual_overlap
                    near_witnesses.append(dict(source_problem_id=source,held_id=identity,**actual_overlap))
                near_by_problem[source]=witnesses
            if observed['complete_question_lexical_copy_candidate_ids']!=list(near_by_problem[source]):
                raise ValueError('Per-presentation lexical flags differ from the complete source-question witnesses')
            if observed['literal_held_stem_matches']:
                text=normalize(tokenizer.decode(actual[1:-1].astype('int64').tolist(),skip_special_tokens=False))
                if hashlib.sha256(text.encode()).hexdigest()!=observed['decoded_normalized_actual_native_text_sha256']:
                    raise ValueError('Actual complete native text fingerprint differs')
                for identity,reported in observed['literal_held_stem_matches'].items():
                    needle=normalize(held[identity]['question']);start=reported['normalized_character_start'];end=start+len(needle)
                    if (text[start:end]!=needle or reported['normalized_character_length']!=len(needle)
                            or ((needle[0].isalnum() or needle[0]=='_') and start and (text[start-1].isalnum() or text[start-1]=='_'))
                            or ((needle[-1].isalnum() or needle[-1]=='_') and end<len(text) and (text[end].isalnum() or text[end]=='_'))):
                        raise ValueError('Actual literal witness and word boundaries do not reconstruct')
                    value=dict(original_presentation_id=row['id'],token_pack_index=position,**reported)
                    observed_problem_literal[source].setdefault(identity,value)
                    literal_witnesses.append(dict(source_problem_id=source,held_id=identity,**value))
            if observed['normalized_held_reference_program_matches']:
                raise ValueError('This completed audit reported zero reference program matches; explicit review needed if changed')
            reasons=[]
            if profile['literal_held_stems']:reasons.append('literal_held_stem_in_actual_native_input')
            if near_by_problem[source]:reasons.append('declared_lexical_question_copy_candidate')
            if profile['normalized_reference_program_matches']:reasons.append('normalized_held_reference_program_match')
            blocked[group].update(reasons)
            counts['native_presentations']+=1;counts['native_context_positions']+=length
            counts['presentations_with_literal_stems']+=bool(observed['literal_held_stem_matches'])
            counts['presentations_with_lexical_question_copy_candidates']+=bool(near_by_problem[source])
            counts['presentations_with_normalized_reference_program_matches']+=bool(observed['normalized_held_reference_program_matches'])
    try:next(ledger);raise ValueError('Unaccounted extra native exposure row')
    except StopIteration:pass
    blocked={k:v for k,v in blocked.items() if v}
    for source,profile in profiles.items():
        if profile['literal_held_stems']!=observed_problem_literal.get(source,{}):raise ValueError('Every source-level literal witness must come from an actual native presentation')
    with gzip.open(DATA/'original-rows.jsonl.gz','rt') as rows:
        for position,line in enumerate(rows):
            row=json.loads(line);entry=pack.index[position]
            if row['group_id'] in blocked:
                expected_exclusions.append(dict(id=row['id'],token_pack_index=position,source_problem_id=row['source_problem_id'],
                    original_split=row['split'],group_id=row['group_id'],reasons=sorted(blocked[row['group_id']])))
            else:
                for key,value in [('presentations',1),('context_positions',int(entry['length'])),
                        ('response_EOS_targets',int(entry['length'])-int(entry['target_start']))]:budgets[row['split']][key]+=value
                if row['split']=='train':expected_positions.append(position)
    exclusions=json.loads((AUDIT/'additional-component-exclusions.json').read_text())
    view=json.loads((AUDIT/'future-training-view-candidate.json').read_text())
    actual_budgets={k:dict(v) for k,v in budgets.items()}
    if (dict(counts)!=complete['actual_counts'] or exclusions['all_excluded_original_presentations']!=expected_exclusions
            or exclusions['component_reasons']!={k:sorted(v) for k,v in blocked.items()}
            or view['original_training_pack_positions']!=expected_positions or view['actual_unflagged_native_budgets']!=actual_budgets
            or actual_budgets!=complete['actual_unflagged_native_budgets']
            or view['original_pack_manifest_sha256']!=file_hash(DATA/'manifest.json')):
        raise ValueError('Every original native input, whole source component exclusion and future view position must reconstruct')
    guard.check(full=True)
    atomic_json(OUT/'all-literal-witnesses.json',literal_witnesses)
    atomic_json(OUT/'all-lexical-copy-witnesses.json',near_witnesses)
    atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
        all67007_actual_native_input_hashes_roles_positions_and_component_view_dispositions_rederived=True,
        actual_native_counts=dict(counts),actual_direct_literal_witnesses=len(literal_witnesses),
        actual_direct_lexical_question_copy_witnesses=len(near_witnesses),actual_additional_excluded_components=len(blocked),
        actual_additional_excluded_presentations=len(expected_exclusions),actual_unflagged_native_budgets=actual_budgets,
        actual_remaining_original_train_positions=len(expected_positions),
        all_reported_positive_witnesses_rebuilt_without_Aho_automaton_or_inverted_candidate_search=True,
        producer_negative_search_not_independently_repeated=True,
        no_original_source_or_held_assessment_case_removed=True,semantic_and_complete_ancestry_independence_not_certified=True,
        source_files_sha256={str(p):file_hash(p) for p in [OUT/'plan.json',OUT/'all-literal-witnesses.json',OUT/'all-lexical-copy-witnesses.json',
            completion_path,AUDIT/'future-training-view-candidate.json',DATA/'manifest.json']},
        dataset_admitted_to_training=False,models_constructed=0,model_updates=0,candidate_promoted=False,goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT/'complete.json'),literal_witnesses=len(literal_witnesses),lexical_witnesses=len(near_witnesses),
        remaining_train_positions=len(expected_positions),admitted_to_training=False)),flush=True)


if __name__=='__main__':main()
