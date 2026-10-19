"""Prune our own run only after two matched large coding regressions.

Never stop on age, missing results or an observation timeout. Retain the
durable second checkpoint and exact decision evidence before any signal.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import time

import assess_joint_java_comparison_v1 as resources
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/fast-joint-regression-monitor-v1'
RUN=ROOT/'reports/fast-joint-v3'


def scores(folder):
    names=['pulse-code-completion.json','pulse-knowledge.json','pulse-math.json','pulse-ocr.json']
    if not all((folder/name).exists() for name in names):return None
    code,knowledge,maths,ocr=[json.loads((folder/n).read_text()) for n in names]
    rows=code['results'];syntax=sum(r['candidate']['syntax_valid'] for r in rows)
    correct=sum(r['candidate']['functional_passed'] for r in rows);efficient=sum(r['efficiency_stress_passed'] for r in rows)
    if len(rows)!=24 or code['task_count']!=24 or round(code['syntax_pass_rate']*24)!=syntax or round(code['functional_pass_at_1']*24)!=correct:
        raise ValueError('Every original code outcome must reconcile before a decision')
    protocol=dict(code=(code['generation_budget'],code['prompt_mode'],code['suite_sha256'],[(r['language'],r['task'],r['specification']) for r in rows]),
        knowledge=(knowledge['generation_budget'],knowledge['questions_sha256']),
        math=(maths['generation_budget'],maths['data_sha256'],maths['evaluated_examples']),
        OCR=(ocr['max_new_tokens'],ocr['data_sha256'],ocr['overall']['examples']),
        precision=[x['inference_precision'] for x in [code,knowledge,maths,ocr]],
        tokenizer=[x['tokenizer_sha256'] for x in [code,knowledge,maths,ocr]])
    return dict(syntax=syntax,correct=correct,efficient=efficient,knowledge_generated=knowledge['overall']['generated_accuracy'],
        math_correct=maths['correct'],OCR_exact=ocr['overall']['exact_match_accuracy'],OCR_CER=ocr['overall']['character_error_rate'],
        protocol=protocol,reports_sha256={str(folder/n):file_hash(folder/n) for n in names})


def regression(base,point):
    if point is None:return False
    if base['protocol']!=point['protocol']:raise ValueError('Only identical development protocols can drive this decision')
    return (point['syntax']<=base['syntax']-5 and point['correct']<=base['correct'] and point['efficient']<=base['efficient']
        and point['knowledge_generated']<=base['knowledge_generated'] and point['math_correct']<=base['math_correct']
        and point['OCR_exact']<=base['OCR_exact'] and point['OCR_CER']>=base['OCR_CER'])


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Monitor must hide every GPU')
    if REPORTS.exists():raise ValueError('Preserve earlier supervisor evidence')
    plan=json.loads((RUN/'plan.json').read_text());identity=plan['owner'];base=scores(RUN/'initial')
    if not base or not resources.same(identity):raise ValueError('Exact live run and complete baseline required')
    REPORTS.mkdir();atomic_json(REPORTS/'plan.json',dict(observed_at=datetime.now(timezone.utc).isoformat(),owner=resources.identity(os.getpid()),
        supervised_exact_owner=identity,source_plan_sha256=file_hash(RUN/'plan.json'),source_sha256=file_hash(__file__),
        baseline=base,checkpoints=[1024,2048],minimum_syntax_pass_loss=5,
        stop_only_after_both_matched_checkpoints_regress=True,never_stop_on_timeout=True,
        preserve_complete_durable_second_checkpoint_before_signal=True,goal_achieved=False))
    deadline=time.monotonic()+4*3600
    while resources.same(identity) and time.monotonic()<deadline:
        first=scores(RUN/'step-1024');second=scores(RUN/'step-2048')
        if first and second:
            decision=regression(base,first) and regression(base,second)
            if not decision:
                atomic_json(REPORTS/'complete.json',dict(action='leave training unchanged',reason='Two-checkpoint large-regression condition not met',
                    first=first,second=second,goal_achieved=False));return
            import torch
            torch.set_num_threads(1)
            current=ROOT/'checkpoints/fast-joint-v3/resume.pt';saved=REPORTS/'retained-resume-step2048.pt'
            os.link(current,saved);state=torch.load(saved,map_location='cpu',weights_only=True,mmap=True)
            if state['completed']!=2048:raise ValueError('Exact durable second checkpoint required before stopping')
            actor=resources.same(identity)
            argv=(Path('/proc')/str(identity['pid'])/'cmdline').read_bytes().split(b'\0');argv=[s.decode() for s in argv if s]
            if not actor or len(argv)!=3 or Path(argv[-1]).name!='train_fast_joint_v3.py' or argv[-2]!='-u':
                raise ValueError('Exact original controller command/birth identity changed; never signal')
            atomic_json(REPORTS/'decision-before-signal.json',dict(at=datetime.now(timezone.utc).isoformat(),action='retire our regressing joint experiment',
                retired_exact_owner=identity,baseline=base,first=first,second=second,durable_actual_updates=2048,
                retained_complete_state=str(saved),retained_native_model=str(ROOT/'checkpoints/fast-joint-v3/step-2048/model.pt'),
                original_source_epoch_and_full_endpoints_explicitly_incomplete=True,
                reason='Large code-syntax loss and no functional/efficiency, knowledge, math or OCR improvement at two matched development checkpoints',
                goal_achieved=False))
            if not resources.same(identity):raise ValueError('Exact controller ended before the signal; do not touch other work')
            os.kill(identity['pid'],signal.SIGTERM)
            until=time.monotonic()+60
            while resources.same(identity) and time.monotonic()<until:time.sleep(.25)
            if resources.same(identity):raise ValueError('Controller still live; inspect exact cleanup without another automatic signal')
            if not (RUN/'restored.json').exists():raise ValueError('Original work restoration not yet proven')
            actors=json.loads((RUN/'children.json').read_text())
            if any(resources.same(a) for a in actors):raise ValueError('Exact registered GPU workers remain live')
            atomic_json(REPORTS/'complete.json',dict(action='regressing experiment retired after exact original-work restoration',
                retained_updates=2048,all_exact_workers_terminal=True,original_work_restored=True,
                better_initial_candidate_and_all_checkpoints_preserved=True,goal_achieved=False));return
        atomic_json(REPORTS/'status.json',dict(at=datetime.now(timezone.utc).isoformat(),exact_run_live=resources.same(identity),
            waiting_for_complete_second_checkpoint_development_results=True,first_checkpoint_available=first is not None,goal_achieved=False))
        time.sleep(5)
    atomic_json(REPORTS/'complete.json',dict(action='no signal',reason='Exact owner ended or monitor observation window expired',
        timeout_is_not_training_failure=True,goal_achieved=False))


if __name__=='__main__':main()
