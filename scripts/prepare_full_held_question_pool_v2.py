"""Reserve every cached pinned MMLU-Pro test question plus existing held QA stems.

No chain-of-thought/reference explanation is copied. New MMLU questions are
separated from the 560 previously sampled IDs. Source-overlap audit is pending.
"""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/full-held-question-pool-v2'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():
        raise ValueError('Fresh CPU-only held question reserve required')
    from datasets import Dataset
    from slm_lab.held_question_registry_v2 import original_question
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    previous=ROOT/'data/knowledge-eval-v1'
    metadata=json.loads((previous/'manifest.json').read_text());revision=metadata['revision']
    source=Path('/home/jovyan/.cache/huggingface/datasets/TIGER-Lab___mmlu-pro/default/0.0.0')/revision/'mmlu-pro-test.arrow'
    dataset=Dataset.from_file(str(source)).select_columns(['question_id','question','options','answer_index','answer','category'])
    seen={str(row['id']) for row in map(json.loads,(previous/'questions.raw.jsonl').read_text().splitlines())}
    if len(seen)!=560 or len(dataset)!=12032:raise ValueError('Exact original cached test split and earlier evaluation sample required')
    rows=[];registry=[]
    for original in dataset:
        identity=str(original['question_id']);options=original['options'];gold=original['answer_index']
        if (not isinstance(original['question'],str) or not original['question'].strip()
                or not 2<=len(options)<=10 or any(not isinstance(s,str) for s in options)
                or type(gold) is not int or not 0<=gold<len(options) or original['answer']!=chr(65+gold)):
            raise ValueError('Every unchanged original text, option order and gold answer index required')
        row=dict(id=identity,domain=original['category'],question=original['question'],options=options,answer_index=gold,
            source=metadata['dataset'],revision=revision,original_split='test',split='test',purpose='evaluation_only',
            optimization_forbidden=True,previously_sampled_for_local_evaluation=identity in seen)
        rows.append(row);registry.append(dict(audit_id='MMLU-Pro:'+identity,benchmark='MMLU-Pro',id=identity,
            question=row['question'],previously_sampled_for_local_evaluation=identity in seen))
    if len({r['id'] for r in rows})!=12032 or not seen<={r['id'] for r in rows}:raise ValueError('All original distinct source IDs required')
    sources=[source,source.with_name('dataset_info.json'),previous/'manifest.json',previous/'questions.raw.jsonl',Path(__file__)]
    for benchmark,name,field in [
        ('GSM8K','data/math-reasoning-v2/test.jsonl','question'),
        ('NQ-open','data/nq-human-questions-v1/valid.jsonl','source_original_question'),
        ('reserved-MBPP','data/verified-code-expanded-v1/reserved-mbpp.jsonl','text')]:
        path=ROOT/name;sources.append(path)
        for row in map(json.loads,path.read_text().splitlines()):
            identity=str(row.get('id',row.get('task_id')))
            registry.append(dict(audit_id=benchmark+':'+identity,benchmark=benchmark,id=identity,
                question=original_question(benchmark,row),previously_sampled_for_local_evaluation=benchmark!='MMLU-Pro'))
    if len({r['audit_id'] for r in registry})!=len(registry):raise ValueError('Distinct benchmark/source IDs required')
    if any(not isinstance(r['question'],str) or not r['question'].strip() for r in registry):raise ValueError('Every original question must be text')
    sources.extend([ROOT/'src/slm_lab/held_question_registry_v2.py', ROOT/'tests/test_held_question_registry_v2.py', ROOT/'reports/held-question-registry-v2-tests.log'])
    for row in registry:
        row.update(split='held',purpose='contamination_audit_only',optimization_forbidden=True)
    OUT.mkdir()
    for name,values in [('mmlu-pro.original-test.jsonl',rows),
                        ('mmlu-pro.not-previously-sampled.raw.jsonl',[r for r in rows if not r['previously_sampled_for_local_evaluation']]),
                        ('question-registry.jsonl',registry)]:
        (OUT/name).write_text(''.join(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n' for row in values))
    atomic_json(OUT/'manifest.json',dict(at=datetime.now(timezone.utc).isoformat(),original_pinned_MMLU_Pro_test_questions=12032,
        previously_sampled_MMLU_questions=560,new_to_local_evaluation_MMLU_questions=11472,
        original_MMLU_domains=dict(Counter(r['domain'] for r in rows)),audit_question_counts=dict(Counter(r['benchmark'] for r in registry)),
        source_revision=revision,license_metadata=metadata['license_metadata'],
        only_question_options_gold_index_exported_no_reference_reasoning=True,
        files_sha256={p.name:file_hash(p) for p in OUT.glob('*.jsonl')},
        source_files_sha256={str(p):file_hash(p) for p in sources},source_overlap_audit_pending=True,
        earlier_or_semantic_training_independence_not_certified=True,optimization_forbidden=True,
        model_generations=0,model_updates=0,candidate_promoted=False,goal_achieved=False))
    print(json.dumps(dict(original_MMLU_questions=12032,new_to_local_evaluation=11472,
        audit_question_counts=dict(Counter(r['benchmark'] for r in registry)),source_audit_pending=True)),flush=True)


if __name__=='__main__':main()
