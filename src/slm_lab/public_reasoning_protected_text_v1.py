"""All declared current held text stems, without consulting gold labels.

This is a bounded input-text overlap inventory, not a semantic independence
certificate or a scan of all historical training bytes.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path

import ahocorasick

from .code_tasks import TASKS
from .held_text_overlap_v1 import HeldTextOverlap,normalize


JSONL_SOURCES=[
    ('full_QA_and_MBPP','data/full-held-question-pool-v2/question-registry.jsonl',17561,'question'),
    ('APPS_original_test','data/public-apps-publisher-test-evaluation-v1/evaluation-records.jsonl',5000,'complete_prompt'),
    ('AI2D_original_test','data/ai2d-all-original-test-eval-v2/questions.jsonl',3088,'question'),
    ('whole_human_preferences','data/whole-human-alignment-v2/preferences.valid.jsonl',4642,'prompt'),
    ('whole_human_dialogue','data/whole-human-alignment-v2/dialogue.valid.jsonl',677,'prompt'),
    ('broad_code_validation','data/broad-evaluation-v1/code.valid.jsonl',113,'text'),
    ('broad_domain_validation','data/broad-evaluation-v1/domains.valid.jsonl',90,'text'),
    ('local_knowledge','data/protected-joint-v1-pulses/knowledge.jsonl',56,'question'),
    ('local_OCR','data/protected-joint-v1-pulses/ocr.jsonl',48,'prompt'),
    ('local_objects','data/protected-joint-v1-pulses/objects.jsonl',15,'prompt'),
    ('local_photos','data/protected-joint-v1-pulses/vision.jsonl',32,'prompt'),
]
SOURCE_FUNCTIONS='reports/source-function-validation-runtime-CPU-v1/all15-fixed-native-validation-cases.json'


def source_paths(root):
    root=Path(root)
    return [root/path for _,path,_,_ in JSONL_SOURCES]+[root/SOURCE_FUNCTIONS,root/'src/slm_lab/code_tasks.py']


def population(root):
    root=Path(root);result=[]
    for family,path,count,field in JSONL_SOURCES:
        # JSONL records end at LF. Unicode paragraph/line separators can occur
        # inside an original question and must not split the JSON string.
        with (root/path).open() as stream:rows=list(map(json.loads,stream))
        if len(rows)!=count:raise ValueError('The complete original protected source population changed: '+family)
        for index,row in enumerate(rows):
            text=row[field]
            if not isinstance(text,str) or not normalize(text):raise ValueError('Complete nonempty original input stem required')
            result.append(dict(audit_id=family+':'+str(index),family=family,question=text,source_path=path,source_position=index))
    fixed=json.loads((root/SOURCE_FUNCTIONS).read_text())
    if len(fixed)!=15 or len(TASKS)!=12:raise ValueError('All original15source validation and12local task specifications required')
    for index,case in enumerate(fixed):
        if case['source_role']!='validation' or case['optimization_forbidden'] is not True:raise ValueError('The original held source role must remain intact')
        result.append(dict(audit_id='source_function_validation:'+str(index),family='source_function_validation',
            question=case['original_source_row']['prompt'],source_path=SOURCE_FUNCTIONS,source_position=index))
    for index,task in enumerate(TASKS):
        name,args,spec=task[:3]
        # Deliberately never read the tuple's reference implementations or tests.
        result.append(dict(audit_id='local_code_specification:'+str(index),family='local_code_specification',
            question=spec,source_path='src/slm_lab/code_tasks.py',source_position=index))
    if len(result)!=31349 or len({r['audit_id'] for r in result})!=31349:raise ValueError('Every declared protected input stem is required')
    return result


def _bounded_pattern(whole,start,end,pattern):
    if (pattern[0].isalnum() or pattern[0]=='_') and start and (whole[start-1].isalnum() or whole[start-1]=='_'):return False
    if (pattern[-1].isalnum() or pattern[-1]=='_') and end+1<len(whole) and (whole[end+1].isalnum() or whole[end+1]=='_'):return False
    return True


def overlaps(candidates,protected,*,checked=lambda:None):
    index=HeldTextOverlap(protected);results={row['id']:[] for row in candidates}
    if len(results)!=len(candidates):raise ValueError('Every source candidate must have a unique original identity')
    mapping=defaultdict(list)
    for position,row in enumerate(candidates):
        if position%256==0:checked()
        text=row['prompt']+'\n'+row['independent_answer']
        for identity,value in index.literal(text).items():
            results[row['id']].append(dict(kind='protected_stem_in_complete_candidate',protected_id=identity,evidence=value))
        for identity,value in index.lexical_copies(text).items():
            results[row['id']].append(dict(kind='rare_word_shingle_copy',protected_id=identity,evidence=value))
        original=normalize(row['original_entry']['question'])
        if not original:raise ValueError('Complete original candidate question required')
        mapping[original].append(row['id'])
    reverse=ahocorasick.Automaton()
    for question,ids in mapping.items():reverse.add_word(question,(question,ids))
    reverse.make_automaton()
    for position,row in enumerate(protected):
        if position%1000==0:checked()
        text=normalize(row['question'])
        for end,(pattern,ids) in reverse.iter(text):
            start=end-len(pattern)+1
            if not _bounded_pattern(text,start,end,pattern):continue
            for identity in ids:results[identity].append(dict(kind='candidate_question_in_protected_input',protected_id=row['audit_id'],
                evidence=dict(normalized_character_start=start,normalized_character_length=len(pattern))))
    return results,dict(index.configuration,complete_protected_counts=dict(Counter(r['family'] for r in protected)),
        reverse_complete_candidate_question_patterns=len(mapping),both_literal_directions=True,
        complete_candidate_prompt_and_response_scanned=True,protected_gold_labels_not_used=True,
        semantic_equivalence_or_historical_exposure_not_certified=True)
