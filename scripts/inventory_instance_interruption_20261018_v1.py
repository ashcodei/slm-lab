"""Read-only inventory of interrupted process births; no signals or training."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from handoff_public_pretraining_lineage_CPU_v1 import STAGES

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/instance-interruption-20261018-v1'

def main():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fresh CPU-only interruption inventory required')
    core=ROOT/'reports/public-pretraining-v1';plan=json.loads((core/'plan.json').read_text())
    births=[plan['owner'],*json.loads((core/'children.json').read_text())]
    files=[Path(__file__),core/'plan.json',core/'children.json',core/'training-metrics.jsonl',core/'progress.json']
    stages=[]
    for _,stage,_,driver in STAGES:
        queue=ROOT/'reports'/(stage+'-queue.json');value=json.loads(queue.read_text());births.append(value['owner']);files.append(queue)
        stages.append(dict(stage=stage,driver=driver,original_queue=str(queue),owner=value['owner'],
            learned_output_exists=(ROOT/'checkpoints'/stage).exists(),reports_exists=(ROOT/'reports'/stage).exists()))
    research=ROOT/'reports/own-function-trajectory-preferences-full-CPU-v2'
    research_plan=research/'plan.json';rplan=json.loads(research_plan.read_text());births.append(rplan['owner']);files.append(research_plan)
    if any(not terminal(row) for row in births):raise ValueError('A captured original process birth still lives; no recovery authorized by inventory')
    rows=[]
    for line in (core/'training-metrics.jsonl').read_bytes().splitlines(keepends=True):
        if not line.endswith(b'\n'):break
        rows.append(json.loads(line))
    if [r['step'] for r in rows]!=list(range(1,len(rows)+1)):raise ValueError('Complete observed update prefix required')
    checkpoints=[dict(path=str(p),bytes=p.stat().st_size) for p in (ROOT/'checkpoints/public-pretraining-v1').rglob('*') if p.is_file()]
    if any(Path(r['path']).name not in ('model.pt','tokenizer.json') for r in checkpoints):
        raise ValueError('Unexpected resumable state needs separate inspection')
    research_states=[dict(path=str(p),bytes=p.stat().st_size) for p in research.rglob('*.pt')]
    OUT.mkdir()
    atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
        current_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        cause='Original process namespace and tmpfs disappeared; physical host reboot not inferred',
        old_registered_births=births,all_old_registered_births_terminal=True,signals_sent=0,
        observed_public_updates=len(rows),durable_public_optimizer_updates=0,
        observed_RAM_only_updates_unrecoverable=len(rows),durable_initial_and_previous_complete_model_intact=True,
        original_public_checkpoints=checkpoints,original_downstream_stages=stages,
        CPU_research_private_states_retained=research_states,
        source_files_sha256={str(p):file_hash(p) for p in files},goal_achieved=False))
    print(json.dumps(dict(receipt=str(OUT/'complete.json'),observed_lost_updates=len(rows),terminal_births=len(births),research_private_states=research_states)))

if __name__=='__main__':main()
