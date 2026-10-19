"""Run the requested correctness/efficiency and knowledge checks after training."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def live(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-pid',type=int,required=True)
    args=p.parse_args();os.chdir(ROOT);reports=ROOT/'reports/posttrain-evaluation-v1';reports.mkdir(exist_ok=True)
    status_path=reports/'status.json'
    def status(state,**fields):
        temporary=status_path.with_suffix('.tmp');temporary.write_text(json.dumps(
            {'status':state,'updated_at':datetime.now(timezone.utc).isoformat(),**fields},indent=2)+'\n');temporary.replace(status_path)
    checkpoint=ROOT/'checkpoints/broad-public-pilot-v1/model.pt'
    training=checkpoint.parent/'training.json'
    status('waiting for completed training',training_pid=args.training_pid)
    while not training.exists():
        if not live(args.training_pid):
            status('training process exited before a completed checkpoint; requires inspection');return
        time.sleep(15)
    report=json.loads(training.read_text())
    plan=json.loads((ROOT/'reports/broad-public-pilot-plan.json').read_text())
    if report['steps']!=plan['optimizer_steps']:raise ValueError('Training stopped short of its planned pass')
    # Let the original loss/generation checks finish before the larger suite.
    while live(args.training_pid) and not (ROOT/'reports/code-generation-after-pilot.json').exists():time.sleep(15)
    questions=ROOT/'data/knowledge-eval-v1/questions.jsonl'
    if not questions.with_suffix('.audit.json').exists():raise ValueError('Knowledge benchmark contamination audit missing')
    for label,command in [
        ('code syntax, correctness and efficiency',[sys.executable,'-u','-m','slm_lab.code_quality',
            '--checkpoint',str(checkpoint),'--output',str(reports/'code-quality.json'),'--device','cuda']),
        ('domain knowledge',[sys.executable,'-u','-m','slm_lab.knowledge_evaluation',
            '--checkpoint',str(checkpoint),'--data',str(questions),'--output',str(reports/'knowledge.json'),'--device','cuda'])]:
        status('evaluating',task=label)
        subprocess.run(command,check=True,cwd=ROOT)
    code=json.loads((reports/'code-quality.json').read_text());knowledge=json.loads((reports/'knowledge.json').read_text())
    summary={'training_updates':report['steps'],'trained_target_tokens':sum(m.get('global_target_tokens',0) for m in report['metrics']),
        'code':{k:code[k] for k in ['task_count','syntax_pass_rate','functional_pass_at_1','efficiency_stress_pass_rate','by_language']},
        'knowledge':{'overall':knowledge['overall'],'by_domain':knowledge['by_domain']},
        'goal_achieved':False,'high_end_llm_parity':'Not established; this suite is a bounded capability assessment.',
        'pending':['Broader languages and repository-level code tasks','Multimodal capability training/evaluation',
                   'Human-feedback policy training and independent quality comparisons','Matched high-end-model comparisons']}
    (reports/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');status('evaluation completed',summary=str(reports/'summary.json'))

if __name__=='__main__':main()
