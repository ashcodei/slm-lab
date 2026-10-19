"""Run fixed code/raw-QA/native-QA assessments sequentially on one capped shared GPU."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    protocol_path=ROOT/'reports/code-instruction-12000-assessment-protocol-v1.json';protocol=json.loads(protocol_path.read_text())
    bound={**protocol['unchanged_evaluation_source_files_sha256'],**protocol['assessment_driver_source_files_sha256'],
        str(protocol_path):file_hash(protocol_path)};guard=FileBindings(bound)
    checkpoint=ROOT/'checkpoints/code-instruction-12000-assessment-v1/model.pt'
    if file_hash(checkpoint)!=protocol['checkpoint_sha256']:raise ValueError('Exact frozen12000-update policy required')
    commands=[('code',[sys.executable,'-u',str(ROOT/'scripts/assess_codenet_model_v2.py'),
        '--checkpoint',str(checkpoint),'--reports','code-instruction-12000-codenet-v1','--device','cuda:0',
        '--expected-stage','sft','--expected-steps','12000','--expected-planned-steps','28343',
        '--expected-data-sha256','e0101ece31a2cbe7b47e41d19741e34602e12e4a5565baea84a33a45f454ca24',
        '--expected-checkpoint-sha256',protocol['checkpoint_sha256'],'--label','own_code_instruction_12000_intermediate_matched_protocol']),
        ('raw_knowledge',[sys.executable,'-u',str(ROOT/'scripts/assess_code_instruction_12000_knowledge_v1.py'),'--protocol','raw']),
        ('native_knowledge',[sys.executable,'-u',str(ROOT/'scripts/assess_code_instruction_12000_knowledge_v1.py'),'--protocol','native'])]
    status_path=ROOT/'reports/code-instruction-12000-assessment-status-v1.json'
    completed=[]
    for name,command in commands:
        guard.check(full=True)
        log=ROOT/'reports'/('code-instruction-12000-'+name+'-v1.log')
        with log.open('xb') as stream:child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
        stat=Path(f'/proc/{child.pid}/stat').read_text();parts=stat[stat.rfind(')')+2:].split()
        receipt=dict(started_at=datetime.now(timezone.utc).isoformat(),phase=name,pid=child.pid,start_ticks=parts[19],argv=command,
            observed_argv=Path(f'/proc/{child.pid}/cmdline').read_bytes().replace(b'\0',b' ').decode().strip(),
            log=str(log.relative_to(ROOT)),protocol_sha256=file_hash(protocol_path),checkpoint_sha256=protocol['checkpoint_sha256'],
            optimizer_updates=0,candidate_promoted=False,goal_achieved=False)
        atomic_json(ROOT/'reports'/('code-instruction-12000-'+name+'-v1-launch.json'),receipt)
        atomic_json(status_path,dict(status='running fixed own-model assessment',updated_at=datetime.now(timezone.utc).isoformat(),
            phase=name,completed_phases=completed,child=receipt,goal_achieved=False))
        code=child.wait()
        if code:
            atomic_json(status_path,dict(status='assessment stopped; preserve partial results and inspect',
                updated_at=datetime.now(timezone.utc).isoformat(),phase=name,exit_code=code,completed_phases=completed,goal_achieved=False))
            raise RuntimeError('Fixed assessment failed: '+name)
        completed.append(name)
    guard.check(full=True)
    atomic_json(status_path,dict(status='all fixed assessments completed; independent matched comparison pending',
        updated_at=datetime.now(timezone.utc).isoformat(),completed_phases=completed,checkpoint_sha256=protocol['checkpoint_sha256'],goal_achieved=False))
    print('all_fixed_12000_update_assessments_completed',flush=True)


if __name__=='__main__':main()
