"""Matched full559-question FP32 assessment of two completed own706M stages.

Read-only native evaluation, with no source/model changes or GPU context.
Earlier CUDA/BF16 scores are not the matched comparison for these CPU runs.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import time

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.knowledge_evaluation import evaluate, letter

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/completed-code-knowledge-CPU-v1'
CHECKPOINTS = {
    'joint': ('checkpoints/extended-curriculum-v1/joint/model.pt', '2dacd490e4a6ed20527a8393629d560e7230c1b6fdec54c1a49dc2eedf7f4e49', 5839),
    'broad-code936': ('checkpoints/dialogue-retention-v1/broad-code/model.pt', '743363d2792c2726352addfe9057649ffa3afe557707e8d82fed0f1bc69f9347', 936),
}
DATA = ROOT / 'data/knowledge-eval-v1/questions.jsonl'


def status(message, **fields):
    atomic_json(REPORTS / 'status.json', dict(status=message, updated_at=datetime.now(timezone.utc).isoformat(),
        model_updates=0, candidate_promoted=False, goal_achieved=False, **fields))


def identity(pid):
    proc = Path('/proc') / str(pid)
    fields = (proc / 'stat').read_text().split(') ', 1)[1].split()
    if fields[0] == 'Z': raise ValueError('Assessment owner is terminal')
    return dict(pid=pid, start_ticks=fields[19], argv_sha256=file_hash(proc / 'cmdline'))


def worker(name):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only native assessment required')
    plan = json.loads((REPORTS / 'plan.json').read_text())
    if identity(plan['owner']['pid']) != plan['owner']:
        raise ValueError('Exact live assessment owner required')
    guard = FileBindings(plan['files_sha256'])
    output = REPORTS / (name + '.json')
    if output.exists(): raise ValueError('Preserve every earlier completed model assessment')
    checkpoint = ROOT / CHECKPOINTS[name][0]
    evaluate(checkpoint, DATA, output, 'cpu', max_examples=559, max_new_tokens=32)
    guard.check(full=True)
    if capture() != plan['runtime']:
        raise ValueError('Native evaluation implementation/runtime changed')


def summarize(path, checkpoint_sha256, rows):
    report = json.loads(path.read_text())
    if (report['checkpoint_sha256'] != checkpoint_sha256 or report['questions_sha256'] != file_hash(DATA)
            or report['generation_budget'] != 32 or report['skipped']
            or report['inference_device_type'] != 'cpu' or report['inference_precision'] != 'fp32'
            or len(report['predictions']) != len(rows) or report['tokenizer_sha256'] != '0072bb6de61da87bd9ba5f4e6739b594cc3e60c62c75a59d25d499629a70088f'):
        raise ValueError('Every fixed question and exact own checkpoint/BPE/CPU protocol required')
    by_domain = {}; generated = likelihood = 0
    for actual, source in zip(report['predictions'], rows):
        scores = actual['candidate_scores']
        g = letter(actual['generated_response'], len(source['options']))
        if len(scores) != len(source['options']) or not all(math.isfinite(v) for v in scores):
            raise ValueError('Complete finite native answer-choice scores required')
        l = max(range(len(scores)), key=lambda i: scores[i])
        if (actual['id'] != source['id'] or actual['domain'] != source['domain']
                or actual['answer_index'] != source['answer_index'] or actual['generated_index'] != g
                or actual['likelihood_index'] != l or actual['generated_correct'] != (g == source['answer_index'])
                or actual['likelihood_correct'] != (l == source['answer_index'])):
            raise ValueError('Raw native responses/likelihoods do not reproduce original scores')
        generated += int(actual['generated_correct']); likelihood += int(actual['likelihood_correct'])
        count = by_domain.setdefault(source['domain'], dict(questions=0, generated_correct=0, likelihood_correct=0))
        count['questions'] += 1; count['generated_correct'] += int(actual['generated_correct']); count['likelihood_correct'] += int(actual['likelihood_correct'])
    if (report['overall']['questions'] != len(rows) or report['overall']['generated_accuracy'] != generated / len(rows)
            or report['overall']['likelihood_accuracy'] != likelihood / len(rows)):
        raise ValueError('Native aggregate scores differ from the full raw predictions')
    for domain, count in by_domain.items():
        if report['by_domain'][domain]['questions'] != count['questions']:
            raise ValueError('Full fixed domain populations required')
    return dict(questions=len(rows), generated_correct=generated, likelihood_correct=likelihood,
        by_domain=by_domain, native_report_sha256=file_hash(path)), report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--worker', choices=list(CHECKPOINTS))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker); return
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or REPORTS.exists():
        raise ValueError('CPU-only fresh owned report directory required; no implicit restart')
    import torch
    torch.set_num_threads(1)
    rows = list(map(json.loads, DATA.read_text().splitlines()))
    if len(rows) != 559 or len({r['id'] for r in rows}) != 559 or len({r['domain'] for r in rows}) != 14:
        raise ValueError('Every original559question/14domain assessment item required')
    runtime = capture()
    files = {str(ROOT / p): h for p, h in runtime['files_sha256'].items()}
    for name, (relative, digest, steps) in CHECKPOINTS.items():
        checkpoint = ROOT / relative; token = checkpoint.parent / 'tokenizer.json'
        if file_hash(checkpoint) != digest:
            raise ValueError('Exact completed own weights required: ' + name)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        if saved['steps'] != steps or saved['planned_steps'] != steps or saved['tokenizer_sha256'] != file_hash(token):
            raise ValueError('Completed own-stage metadata and own tokenizer required: ' + name)
        del saved
        files[str(checkpoint)] = digest; files[str(token)] = file_hash(token)
    for path in [Path(__file__).resolve(), DATA, DATA.with_suffix('.audit.json'),
            ROOT / 'reports/completed-broad-code-state-readback-v1.json']:
        files[str(path)] = file_hash(path)
    guard = FileBindings(files)
    REPORTS.mkdir()
    plan = dict(recorded_at=datetime.now(timezone.utc).isoformat(), owner=identity(os.getpid()),
        checkpoints={n: str(ROOT / value[0]) for n, value in CHECKPOINTS.items()}, files_sha256=files, runtime=runtime,
        questions=559, domains=dict(Counter(r['domain'] for r in rows)), generation_budget=32,
        device='cpu', precision='fp32', torch_threads_per_child=2,
        original_native_prompt_parser_choice_scores_and_generation_unchanged=True,
        legacy_native_protocol_note='The native report literal says BF16; sealed CPU/fp32 metadata and this matched plan describe the actual arithmetic.',
        contamination_scope='Reused public/development sample. Earlier exact-text audit does not certify no subsequent exposure or semantic overlap.',
        matched_against_earlier_CUDA_BF16_scores=False, model_updates=0, candidate_promoted=False, goal_achieved=False)
    atomic_json(REPORTS / 'plan.json', plan)
    children = {}; streams = []
    try:
        for name in CHECKPOINTS:
            stream = (REPORTS / (name + '.log')).open('x'); streams.append(stream)
            child = subprocess.Popen(['.venv/bin/python', '-u', str(Path(__file__).resolve()), '--worker', name],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false'})
            children[name] = child
        status('evaluating both completed own706M stages on all559fixed questions; CPU-only',
            workers={n: identity(p.pid) for n, p in children.items()})
        while any(p.poll() is None for p in children.values()):
            guard.check()
            if any(p.poll() not in [None, 0] for p in children.values()):
                raise ValueError('Native assessment child stopped; inspect retained raw logs')
            time.sleep(5)
        if any(p.returncode != 0 for p in children.values()):
            raise ValueError('Native assessment child did not complete')
        summaries = {}; native = {}
        for name, (_, digest, _) in CHECKPOINTS.items():
            summaries[name], native[name] = summarize(REPORTS / (name + '.json'), digest, rows)
        if native['joint']['evaluation_provenance'] != native['broad-code936']['evaluation_provenance'] or capture() != runtime:
            raise ValueError('Exact common native evaluator/runtime required')
        guard.check(full=True)
        changes = {}
        for metric in ['generated_correct', 'likelihood_correct']:
            a, b = native['joint']['predictions'], native['broad-code936']['predictions']
            changes[metric] = dict(gained_ids=[x['id'] for x, y in zip(a, b) if not x[metric] and y[metric]],
                lost_ids=[x['id'] for x, y in zip(a, b) if x[metric] and not y[metric]])
        atomic_json(REPORTS / 'comparison.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
            plan_sha256=file_hash(REPORTS / 'plan.json'), summaries=summaries, paired_changes=changes,
            comparison_scope='Matched native full559question CPU/fp32 descriptive stage comparison. Public/development data, not certified independent final benchmark or causal isolation.',
            prior_CUDA_scores_not_matched_controls=True, model_updates=0, candidate_promoted=False, goal_achieved=False))
        status('matched full559question CPU assessments completed; inspect measured results', summaries=summaries)
    except BaseException as error:
        status('CPU assessment stopped; inspect retained evidence before any restart', error=str(error))
        for child in children.values():
            if child.poll() is None: child.terminate()
        for child in children.values():
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=10)
        raise
    finally:
        for stream in streams: stream.close()


if __name__ == '__main__': main()
