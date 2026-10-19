"""Read-only CPU 2x2 model/frame diagnosis after native SFT knowledge regression."""
from datetime import datetime, timezone
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/knowledge-chat-diagnostic-v1'
SFT = ROOT/'reports/whole-human-SFT-v1'
CHECKPOINTS = ROOT/'checkpoints/whole-human-SFT-v1'
QUESTIONS = ROOT/'data/knowledge-eval-v1/questions.jsonl'


def identity(pid):
    try:
        values = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, ticks=values[19], state=values[0]) if values[0] != 'Z' else None
    except (OSError, IndexError): return None


def same(actor):
    found = identity(actor['pid'])
    return found if found and found['ticks'] == actor['ticks'] else None


def worker(subject):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU diagnostic must explicitly hide every CUDA device')
    plan = json.loads((REPORTS/'plan.json').read_text()); me = identity(os.getpid())
    def owned():
        if not same(plan['owner']) or time.time() > plan['expires_at_unix']:
            raise ValueError('Exact live CPU owner and bounded diagnostic lifetime required')
    while True:
        owned()
        if any(a['pid'] == me['pid'] and a['ticks'] == me['ticks'] and a['subject'] == subject
               for a in json.loads((REPORTS/'children.json').read_text())): break
        time.sleep(.05)
    import torch
    from tokenizers import Tokenizer
    from slm_lab.knowledge_chat_diagnostic_v1 import evaluate_rows
    from slm_lab.whole_human_sft_inference_v1 import load
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    guard = FileBindings({**plan['source_files_sha256'], str(REPORTS/'plan.json'):file_hash(REPORTS/'plan.json')})
    spec = plan['subjects'][subject]; checkpoint = CHECKPOINTS/spec['arm']/'model.pt'
    model, _ = load(checkpoint, SFT/'plan.json', spec['arm'], expected_checkpoint_sha256=spec['checkpoint_sha256'],
                    expected_plan_sha256=plan['SFT_plan_sha256'], device='cpu')
    tok = Tokenizer.from_file(str(CHECKPOINTS/spec['arm']/'tokenizer.json'))
    rows = [json.loads(line) for line in QUESTIONS.read_text().splitlines()]
    signatures = [(id(p), p._version, p.data_ptr()) for p in model.parameters()]
    with (REPORTS/(subject+'-raw-tokens.jsonl')).open('x') as journal:
        count = [0]
        def checked(): owned(); guard.check()
        def record(row):
            journal.write(json.dumps(row, ensure_ascii=False)+'\n'); journal.flush(); count[0] += 1
            if count[0]%20 == 0: print(subject, count[0], '/', len(rows), flush=True)
        result = evaluate_rows(model.aligned, tok, rows, spec['frame'], max_new_tokens=32, checked=checked, record=record)
    if len(result['predictions']) != 559 or signatures != [(id(p), p._version, p.data_ptr()) for p in model.parameters()]:
        raise ValueError('Every original knowledge question and all fixed model versions/pointers required')
    result.update(checkpoint_sha256=spec['checkpoint_sha256'], run_plan_sha256=file_hash(REPORTS/'plan.json'),
        questions_sha256=plan['questions_sha256'], tokenizer_sha256=file_hash(CHECKPOINTS/spec['arm']/'tokenizer.json'),
        raw_journal_sha256=file_hash(REPORTS/(subject+'-raw-tokens.jsonl')), actual_counts=model.counts(), model_updates=0)
    guard.check(full=True); atomic_json(REPORTS/(subject+'.json'), result)


def reconcile(plan):
    from tokenizers import Tokenizer
    from slm_lab.knowledge_chat_diagnostic_v1 import question_prompt, summarize
    from slm_lab.knowledge_evaluation import letter
    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines()]; summaries = {}
    for subject, spec in plan['subjects'].items():
        result = json.loads((REPORTS/(subject+'.json')).read_text())
        raw_path = REPORTS/(subject+'-raw-tokens.jsonl')
        raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
        tok = Tokenizer.from_file(str(CHECKPOINTS/spec['arm']/'tokenizer.json'))
        if (len(raw) != len(questions) or result['predictions'] != raw or result['raw_journal_sha256'] != file_hash(raw_path)
                or result['checkpoint_sha256'] != spec['checkpoint_sha256'] or result['run_plan_sha256'] != file_hash(REPORTS/'plan.json')):
            raise ValueError('Complete actual source-bound raw deliveries and corresponding diagnostic result required')
        for question, row in zip(questions, raw):
            prompt = question_prompt(question, spec['frame']); answer = tok.decode(row['generated_ids'])
            prediction = letter(answer, len(question['options'])); candidates = row['candidate_token_evidence']
            scores = [sum(v['per_token_logprobs']) for v in candidates]; best = max(range(len(scores)), key=lambda i:scores[i])
            if (row['id'] != question['id'] or row['domain'] != question['domain'] or row['answer_index'] != int(question['answer_index'])
                    or row['prompt'] != prompt or row['prompt_ids'] != [1, *tok.encode(prompt).ids]
                    or not 1 <= len(row['generated_ids']) <= 32 or row['context_tokens'] != len(row['prompt_ids'])
                    or row['generated_response'] != answer or row['generated_index'] != prediction
                    or row['generated_correct'] != (prediction == row['answer_index'])
                    or scores != row['candidate_scores'] or row['likelihood_index'] != best or row['likelihood_correct'] != (best == row['answer_index'])
                    or len(candidates) != len(question['options'])
                    or any(not math.isfinite(value) or value > 0 for item in candidates for value in item['per_token_logprobs'])
                    or any(v['token_ids'] != tok.encode(' '+chr(65+i)).ids or len(v['token_ids']) != len(v['per_token_logprobs']) for i,v in enumerate(candidates))):
                raise ValueError('Every whole original prompt/choice/generated token delivery and parsed score must reconstruct')
        if summarize(raw) != result['overall']: raise ValueError('Every original diagnostic score must aggregate from raw evidence')
        summaries[subject] = result['overall']
    atomic_json(REPORTS/'complete.json', dict(matched_CPU_FP32_model_and_prompt_factorial=summaries,
        all2236_original_question_frame_policy_deliveries_and_scores_reconciled=True,
        numerical_CPU_forwards_not_independently_replayed=True, post_hoc_diagnostic=True,
        native_benchmark_results_not_replaced=True, model_updates=0, candidate_promoted=False, goal_achieved=False))


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU parent requires hidden CUDA devices')
    if REPORTS.exists(): raise ValueError('Preserve earlier diagnostic attempts; no automatic restart')
    if '1 passed' not in (ROOT/'reports/knowledge-chat-diagnostic-v1-tests.log').read_text():
        raise ValueError('Actual native numerical parity/raw-token/source-frame check required')
    old = json.loads((SFT/'plan.json').read_text()); selection = json.loads((SFT/'selection.json').read_text())
    selected = selection['selected_arm']
    if selected == 'initial': raise ValueError('Diagnostic requires an actual trained selected arm')
    paths = ['scripts/assess_knowledge_chat_diagnostic_v1.py', 'src/slm_lab/knowledge_chat_diagnostic_v1.py',
             'tests/test_knowledge_chat_diagnostic_v1.py', 'reports/knowledge-chat-diagnostic-v1-tests.log']
    files = {str(ROOT/name):file_hash(ROOT/name) for name in paths}
    files.update({path:sha for path,sha in old['source_files_sha256'].items() if '/src/slm_lab/' in path})
    for path in (SFT/'plan.json', SFT/'selection.json', QUESTIONS, QUESTIONS.with_suffix('.audit.json')):
        files[str(path)] = file_hash(path)
    subjects = {}
    for arm in ('initial', selected):
        for path in (CHECKPOINTS/arm/'model.pt', CHECKPOINTS/arm/'tokenizer.json', SFT/arm/'knowledge.json'):
            files[str(path)] = file_hash(path)
        for frame in ('plain', 'chat'):
            subjects[arm+'-'+frame] = dict(arm=arm, frame=frame, checkpoint_sha256=file_hash(CHECKPOINTS/arm/'model.pt'))
    guard = FileBindings(files); guard.check(full=True); REPORTS.mkdir()
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), owner=identity(os.getpid()),
        source_files_sha256=files, SFT_plan_sha256=file_hash(SFT/'plan.json'), questions_sha256=file_hash(QUESTIONS),
        subjects=subjects, expires_at_unix=time.time()+12*3600, CPU_only=True, threads_per_worker=2,
        generation_budget=32, original_GPU_work_never_signaled=True, automatic_retry=False,
        post_hoc_diagnostic=True, candidate_promoted=False, goal_achieved=False)
    atomic_json(REPORTS/'plan.json', plan); atomic_json(REPORTS/'children.json', [])
    children = []; actors = []; driver = str(Path(__file__).resolve())
    try:
        for subject in subjects:
            with (REPORTS/(subject+'.log')).open('x') as stream:
                child = subprocess.Popen(['.venv/bin/python', '-u', driver, '--worker', subject], cwd=ROOT,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2', 'TOKENIZERS_PARALLELISM':'false'})
            actor = identity(child.pid)
            if not actor: raise ValueError('Exact CPU child identity missing before registration')
            children.append(child); actors.append({**actor, 'subject':subject}); atomic_json(REPORTS/'children.json', actors)
        while any(child.poll() is None for child in children):
            if time.time() > plan['expires_at_unix'] or any(child.poll() not in (None, 0) for child in children):
                raise ValueError('Bounded CPU diagnostic child failed or expired; preserve evidence without retry')
            time.sleep(3)
        if any(child.returncode != 0 for child in children): raise ValueError('Every CPU diagnostic child must complete')
        guard.check(full=True); reconcile(plan)
    except BaseException as error:
        atomic_json(REPORTS/'failure.json', dict(error=repr(error), automatic_retry=False, goal_achieved=False))
        for actor in actors:
            if same(actor): os.kill(actor['pid'], signal.SIGTERM)
        for child in children:
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                actor = next(a for a in actors if a['pid'] == child.pid)
                if same(actor): os.kill(child.pid, signal.SIGKILL)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--worker'); args = parser.parse_args()
    if args.worker: worker(args.worker)
    else: parent()
