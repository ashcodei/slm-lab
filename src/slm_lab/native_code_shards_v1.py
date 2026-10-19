"""Distribute unchanged greedy code cases; reconcile every original case.

This changes scheduling only. Prompts,512-token budgets, workers and cases stay
unchanged. Raw per-shard reports remain separate from the complete merge.
"""
import json
from pathlib import Path

from .code_tasks import TASKS
from .corpus import file_hash
from .diverse_images import atomic_json

FORMAT='own-native-code-task-shard-v1'
LANGUAGES=('Python','JavaScript')


def positions(rank,world):
    if type(world) is not int or not 1<=world<=len(TASKS) or type(rank) is not int or not 0<=rank<world:
        raise ValueError('Nonempty explicit original task shard required')
    return list(range(rank,len(TASKS),world))


def evaluate_shard(model,metadata,checkpoint,folder,*,rank,world,frame,device='cuda:0',budget=512):
    from . import code_quality
    if frame not in ('completion','instruction') or type(budget) is not int or budget!=512:
        raise ValueError('Unchanged full native code frame and512-token budget required')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);selected=positions(rank,world)
    raw=folder/f'code-{frame}-shard-{rank}.raw.json';output=folder/f'code-{frame}-shard-{rank}.json'
    if raw.exists() or output.exists():raise ValueError('Preserve previous raw and enveloped assessments')
    original_tasks,original_loader=code_quality.TASKS,code_quality.load_checkpoint
    try:
        code_quality.TASKS=[TASKS[i] for i in selected];code_quality.load_checkpoint=lambda *args,**kwargs:(model,metadata)
        code_quality.evaluate(checkpoint,raw,device=device,max_new_tokens=budget,languages=LANGUAGES,prompt_mode=frame)
    finally:code_quality.TASKS=original_tasks;code_quality.load_checkpoint=original_loader
    atomic_json(output,dict(format=FORMAT,rank=rank,world=world,original_task_positions=selected,prompt_mode=frame,
        generation_budget=budget,raw_report=str(raw.resolve()),raw_report_sha256=file_hash(raw),
        sharding_implementation_sha256=file_hash(__file__),greedy_cases_have_no_cross_case_training_or_sampling=True,goal_achieved=False))
    return output


def reconcile(envelopes,output):
    envelopes=[Path(p) for p in envelopes];output=Path(output)
    if output.exists() or not envelopes:raise ValueError('Fresh complete assessment destination required')
    rows={};shared=None;inventory={};world=len(envelopes);seen_ranks=set()
    for path in envelopes:
        envelope=json.loads(path.read_text());rank=envelope.get('rank');raw=Path(envelope['raw_report'])
        if (envelope.get('format')!=FORMAT or envelope.get('world')!=world or rank in seen_ranks
                or envelope.get('original_task_positions')!=positions(rank,world)
                or file_hash(raw)!=envelope['raw_report_sha256'] or envelope.get('sharding_implementation_sha256')!=file_hash(__file__)):
            raise ValueError('Every exact original shard identity and raw report byte required')
        report=json.loads(raw.read_text());profile={k:report[k] for k in ['checkpoint_sha256','generation_budget','decoding','prompt_mode',
            'suite_sha256','tokenizer_sha256','inference_device_type','inference_precision','evaluation_provenance','limitations']}
        if shared is None:shared=profile
        if (shared!=profile or profile['suite_sha256']!=file_hash(Path(__file__).with_name('code_tasks.py'))
                or profile['generation_budget']!=512 or profile['generation_budget']!=envelope['generation_budget']
                or profile['prompt_mode']!=envelope['prompt_mode']):raise ValueError('All original prompts/weights/budgets/runtime must match')
        expected=[(lang,TASKS[i][0]) for lang in LANGUAGES for i in positions(rank,world)]
        if [(r['language'],r['task']) for r in report['results']]!=expected or report['task_count']!=len(expected):
            raise ValueError('Every original assigned code case must appear exactly once in order')
        checks=dict(syntax_pass_rate=sum(r['candidate'].get('syntax_valid',False) for r in report['results'])/len(expected),
            functional_pass_at_1=sum(r['candidate'].get('functional_passed',False) for r in report['results'])/len(expected),
            efficiency_stress_pass_rate=sum(r['efficiency_stress_passed'] for r in report['results'])/len(expected))
        if any(report[k]!=v for k,v in checks.items()):raise ValueError('Every actual shard aggregate must reconstruct from its case outputs')
        for row in report['results']:
            key=(row['language'],row['task'])
            if key in rows or row['efficiency_stress_passed'] and not row['candidate'].get('functional_passed'):
                raise ValueError('No repeated or impossible favorable case verdict allowed')
            rows[key]=row
        seen_ranks.add(rank);inventory[str(path.resolve())]=file_hash(path);inventory[str(raw)]=file_hash(raw)
    if seen_ranks!=set(range(world)):raise ValueError('Every rank required')
    order=[(lang,t[0]) for lang in LANGUAGES for t in TASKS]
    if set(rows)!=set(order):raise ValueError('Complete original local suite must reconcile; no omitted failures')
    results=[rows[k] for k in order];n=len(results)
    report=dict(shared,task_count=n,results=results,
        syntax_pass_rate=sum(r['candidate'].get('syntax_valid',False) for r in results)/n,
        functional_pass_at_1=sum(r['candidate'].get('functional_passed',False) for r in results)/n,
        efficiency_stress_pass_rate=sum(r['efficiency_stress_passed'] for r in results)/n,
        by_language={lang:dict(tasks=sum(r['language']==lang for r in results),
            syntax_passed=sum(r['language']==lang and r['candidate'].get('syntax_valid',False) for r in results),
            functional_passed=sum(r['language']==lang and r['candidate'].get('functional_passed',False) for r in results),
            efficiency_passed=sum(r['language']==lang and r['efficiency_stress_passed'] for r in results)) for lang in LANGUAGES},
        exact_all_original_cases_reconciled=True,raw_reports_preserved=True,shards=world,source_files_sha256=inventory,
        sharding_implementation_sha256=file_hash(__file__),independent_code_reexecution_still_required=True,goal_achieved=False)
    atomic_json(output,report);return report
