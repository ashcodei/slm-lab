"""Paired diagnostic of unchanged generated Python under one added builtin.

All original benchmark artifacts, prompts, candidates, cases and scores stay
unchanged. This separate CPU counterfactual adds map only; its results are not
a new official score or training targets.
"""
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'reports/code-map-runtime-diagnostic-v1.json'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('The diagnostic must run on CPU with every GPU hidden')
    if OUTPUT.exists():raise ValueError('Preserve prior diagnostic attempts')
    from slm_lab import code_quality
    from slm_lab.code_tasks import TASKS,request
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    if '1 passed' not in (ROOT/'reports/map-runtime-diagnostic-v1-tests.log').read_text():raise ValueError('Actual differential map/import guard check required')
    original_worker=code_quality.WORKER;fork=ROOT/'scripts/code_worker_map_diagnostic_v1.py';rows=[];sources={}
    tasks={t[0]:t for t in TASKS};reports=ROOT/'reports/code-initialized-alignment-v1'
    required=[reports/a/('code-'+m+'.json') for a in ('initial','mixed_CE','mixed_reference_KL') for m in ('completion','instruction')]
    actor=next(a for a in json.loads((reports/'children.json').read_text()) if a['subject']=='assess-mixed_reference_KL-code-1')
    deadline=time.monotonic()+3600
    while not all(p.exists() for p in required):
        try:stat=(Path('/proc')/str(actor['pid'])/'stat').read_text().rsplit(') ',1)[1].split()
        except FileNotFoundError:raise ValueError('Exact code worker ended without all original code reports')
        if stat[19]!=actor['ticks'] or stat[0] in ('Z','X'):raise ValueError('Exact code worker terminal without complete reports')
        if time.monotonic()>deadline:raise ValueError('Exact code worker remains live beyond diagnostic observation; do not restart evaluation')
        time.sleep(5)
    for arm in ('initial','mixed_CE','mixed_reference_KL'):
        for mode in ('completion','instruction'):
            path=reports/arm/('code-'+mode+'.json')
            if not path.exists():raise ValueError('Every original all-arm code report must exist before the paired diagnostic')
            source=json.loads(path.read_text());sources[str(path)]=file_hash(path)
            candidates=[r for r in source['results'] if r['language']=='Python']
            if len(candidates)!=12 or {r['task'] for r in candidates}!=set(tasks):raise ValueError('Every original Python task required')
            for row in candidates:
                task=tasks[row['task']]
                if row['specification']!=task[2]:raise ValueError('Keep every original specification unchanged')
                payload=request(task,'Python',row['code'])
                code_quality.WORKER=original_worker;before=code_quality.execute(payload)
                code_quality.WORKER=fork;after=code_quality.execute(payload)
                if (before.get('syntax_valid')!=row['candidate'].get('syntax_valid')
                        or before.get('functional_passed')!=row['candidate'].get('functional_passed')):
                    raise ValueError('Original preserved source/runner must reproduce original syntax and functional score')
                rows.append(dict(arm=arm,prompt_mode=mode,task=row['task'],unchanged_code=row['code'],
                    original_runtime=before,only_map_added=after,original_case_count=len(payload['cases']),
                    code_sha256=__import__('hashlib').sha256(row['code'].encode()).hexdigest()))
            print('complete_original_Python_pair',arm,mode,flush=True)
    code_quality.WORKER=original_worker
    for path,value in sources.items():
        if file_hash(path)!=value:raise ValueError('Every original native report must remain unchanged')
    summary={}
    for arm in ('initial','mixed_CE','mixed_reference_KL'):
        summary[arm]={}
        for mode in ('completion','instruction'):
            chosen=[r for r in rows if r['arm']==arm and r['prompt_mode']==mode]
            summary[arm][mode]=dict(original_Python_tasks=len(chosen),original_functional=sum(r['original_runtime'].get('functional_passed',False) for r in chosen),
                diagnostic_functional=sum(r['only_map_added'].get('functional_passed',False) for r in chosen),
                original_errors=dict(Counter(r['original_runtime'].get('error','passed') for r in chosen)),
                diagnostic_errors=dict(Counter(r['only_map_added'].get('error','passed') for r in chosen)))
    atomic_json(OUTPUT,dict(observed_at=datetime.now(timezone.utc).isoformat(),source_reports_sha256=sources,
        original_worker_sha256=file_hash(original_worker),diagnostic_worker_sha256=file_hash(fork),reader_sha256=file_hash(Path(__file__)),
        only_runtime_change='Add the pure Python map builtin to the existing restricted namespace; all AST/import/resource restrictions retained',
        every_original_Python_task_paired=rows,summary=summary,official_scores_unchanged=True,model_updates=0,
        this_diagnostic_not_training_data_or_proof_of_generic_Python_correctness=True,candidate_promoted=False,goal_achieved=False))
    print(OUTPUT,flush=True)


if __name__=='__main__':main()
