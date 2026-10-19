"""Seal complete Java generations, then run every original published module suite."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil

import torch
from tokenizers import Tokenizer

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .evaluation_provenance import capture, seal
from .exercise_java_runner import PublishedSuites, extract_files, inventory, HARNESS, LAUNCHER
from .recurrent_ablation import state_digest
from .training import Encoder, load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = 'reports/exercism-java-publisher-references-v3/summary.json'
PROTOCOL = ('All35 original held Java modules; original instructions plus exact publisher-provided supporting '
    'files and requested reference-bearing solution paths; all complete raw greedy generations sealed before '
    'published-case execution; up to2048 new tokens; original JUnit5.10.0/4.13 and AssertJ versions, all teaching '
    'disabled cases enabled without test edits; pre-exec confined JVM; exact reference-matched test inventories.')


def question(row, suite):
    text = row['prompt']
    if suite['editors']:
        text += '\n\nThese original publisher-provided supporting files are available unchanged. Do not return replacements:\n'
        for name, raw in sorted(suite['editors'].items()):
            text += ('\nFile: ' + name + '\n```java\n' + raw.decode()
                + ('' if raw.endswith(b'\n') else '\n') + '```\n')
    text += ('\n\nReturn only the complete requested solution files, each labeled File: path followed by a java code block. '
        'Requested solution files:\n' + '\n'.join(suite['expected_solution_files']))
    return {**row, 'prompt': text}


def case_layout(row, suite, encoder):
    prompt = encoder.prompt(question(row, suite))
    tokens = [encoder.bos] + encoder.tokens(prompt)
    budget = min(2048, encoder.config.max_text_length - len(tokens))
    if budget < len(encoder.tokens(row['response'])) + 1:
        raise ValueError('Complete original Java prompt/target does not fit the unchanged generation context: ' + row['id'])
    return prompt, tokens, budget


def _case_pass(row):
    try:
        extract_files(row['raw_response'], row['expected_solution_files'])
        delivered = True
    except ValueError:
        delivered = False
    execution = row.get('execution', {})
    compiled = delivered and execution.get('returncode') == 0 and execution.get('limit') is None and bool(execution)
    passed = False
    if compiled:
        runtime = execution.get('execution', {})
        raw = execution.get('raw_inventory_output')
        if raw is not None:
            measured = inventory(raw.encode())
            if measured != execution.get('inventory') or hashlib.sha256(raw.encode()).hexdigest() != execution.get('report_sha256'):
                raise ValueError('Java inventory differs from retained actual JVM output')
            passed = (runtime.get('returncode') == 0 and runtime.get('limit') is None
                and measured['cases'] == row['expected_registered_cases'] and measured['all_passed'])
    if (row['full_file_delivery'], row['compilation_passed'], row['published_tests_passed']) != (delivered, compiled, passed):
        raise ValueError('Java case flags differ from delivered source/compiler/JVM evidence')
    return passed


def compare_reports(before, after):
    for key in ['evaluation_provenance', 'data_sha256', 'manifest_sha256', 'tokenizer_sha256',
            'source_archive_sha256', 'Java_toolchain_manifest_sha256', 'test_framework_manifest_sha256',
            'reference_report_sha256', 'declared_generation_budget', 'protocol', 'inference_precision', 'inference_device_type']:
        if before[key] != after[key]:
            raise ValueError('Unmatched complete Java assessment: ' + key)
    identity = []
    for report in [before, after]:
        rows = report['results']
        if len(rows) != 35 or len({r['id'] for r in rows}) != 35:
            raise ValueError('All35original Java modules required')
        for row in rows:
            _case_pass(row)
        for metric, field in [('full_file_delivery_rate', 'full_file_delivery'),
                ('compilation_success_rate', 'compilation_passed'), ('published_module_pass_rate', 'published_tests_passed')]:
            if report[metric] != sum(r[field] for r in rows) / 35:
                raise ValueError('Java aggregate differs from complete source/test case evidence')
        identity.append([(r['id'], r['group_id'], r['prompt_sha256'], r['generation_budget'],
            r['original_test_sha256'], r['original_editor_sha256'], r['expected_solution_files'],
            r['expected_registered_cases']) for r in rows])
    if identity[0] != identity[1]:
        raise ValueError('Unmatched complete original Java prompts/support/test inventories')
    lost = [a['id'] for a, b in zip(before['results'], after['results']) if a['published_tests_passed'] and not b['published_tests_passed']]
    return dict(passed=not lost and all(after[k] >= before[k] for k in
            ['full_file_delivery_rate', 'compilation_success_rate', 'published_module_pass_rate']),
        previously_correct_modules_lost=lost, before_module_pass_rate=before['published_module_pass_rate'],
        after_module_pass_rate=after['published_module_pass_rate'],
        base_checkpoint_sha256=before['checkpoint_sha256'], candidate_checkpoint_sha256=after['checkpoint_sha256'],
        goal_achieved=False, scope='Matched complete Java publisher development suites; retention is not all-code correctness, efficiency optimality or frontier capability.')


@torch.inference_mode()
def evaluate(checkpoint, data, output, device='cuda:2'):
    checkpoint, data, output = Path(checkpoint), Path(data), Path(output)
    evidence = output.with_name(output.stem + '-evidence')
    if output.exists() or evidence.exists():
        raise ValueError('Preserve existing Java generations/results; use a new output')
    if shutil.disk_usage(ROOT).free < 26 * 1024**3 + 64 * 1024**2:
        raise ValueError('Reserve original26GiB and64MiB complete Java reports')
    manifest_path = data.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    data_sha, manifest_sha = file_hash(data), file_hash(manifest_path)
    if data.name != 'valid.jsonl' or manifest['files_sha256'][data.name] != data_sha:
        raise ValueError('Exact finalized original Java held source required')
    ref_path = ROOT / REFERENCE
    refs = json.loads(ref_path.read_text())
    if (refs.get('complete_all35_reference_suites_passed') is not True or refs.get('passed') != 35
            or refs.get('data_sha256') != data_sha or refs.get('manifest_sha256') != manifest_sha):
        raise ValueError('All35 original Java references must first pass complete publisher tests')
    FileBindings(refs['source_files_sha256']).check(full=True)
    for key, path in [('Java_toolchain_manifest_sha256', ROOT / '.tools/java-toolchain-v1.json'),
            ('test_framework_manifest_sha256', ROOT / '.tools/java-test-framework-v1.json')]:
        if file_hash(path) != refs[key]:
            raise ValueError('Pinned verified Java compiler/framework changed')
    files = list(dict.fromkeys([ROOT / p for p in capture()['files_sha256']] + [Path(__file__),
        ROOT / 'src/slm_lab/recurrent_ablation.py', ref_path] + [Path(p) for p in refs['source_files_sha256']]))
    runtime = capture(files)
    rows = [r for r in jsonl(data) if r['language'] == 'Java']
    references = {r['id']: r for r in refs['results']}
    if len(rows) != 35 or set(references) != {r['id'] for r in rows}:
        raise ValueError('Complete35original Java module population differs')
    guard = FileBindings({**{str(p): file_hash(p) for p in files},
        str(checkpoint): file_hash(checkpoint), str(checkpoint.parent / 'tokenizer.json'): file_hash(checkpoint.parent / 'tokenizer.json')})
    suites = PublishedSuites(ROOT / 'data/exercism-source-cache-v1/java')
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence.mkdir()
    try:
        torch.set_num_threads(1)
        model, metadata = load_checkpoint(checkpoint, device, training_state=False)
        model.eval()
        if sum(p.numel() for p in model.parameters()) >= 10**9:
            raise ValueError('Exact own sub1B policy required')
        token = checkpoint.parent / 'tokenizer.json'
        if file_hash(token) != manifest['tokenizer_sha256'] or metadata['tokenizer_sha256'] != manifest['tokenizer_sha256']:
            raise ValueError('Original Java data and own policy tokenizer differ')
        encoder = Encoder(Tokenizer.from_file(str(token)), model.config, data.parent)
        layouts = []
        for row in rows:
            suites.reference(row)
            suite = suites.suite(row)
            reference = references[row['id']]
            if (suite['test_sha256'] != reference['original_test_sha256']
                    or suite['editor_sha256'] != reference['original_editor_sha256']):
                raise ValueError('Original complete Java test/support bytes differ from verified references')
            prompt, tokens, budget = case_layout(row, suite, encoder)
            layouts.append((row, suite, reference, prompt, tokens, budget))
        before = state_digest(model.state_dict())
        amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
        generated = []
        path = evidence / 'generations.jsonl'
        with path.open('x') as stream:
            for row, suite, reference, prompt, tokens, budget in layouts:
                guard.check()
                ids = torch.tensor([tokens], device=device)
                with amp():
                    value = model.generate(ids, max_new_tokens=budget, eos_token_id=encoder.eos, use_cache=True)
                record = dict(id=row['id'], group_id=row['group_id'], prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    prompt_tokens=len(tokens), generation_budget=budget,
                    expected_solution_files=sorted(suite['expected_solution_files']), original_test_sha256=suite['test_sha256'],
                    original_editor_sha256=suite['editor_sha256'], expected_registered_cases=reference['result']['inventory']['cases'],
                    raw_response=encoder.tokenizer.decode(value[0,len(tokens):].tolist()),
                    generated_token_ids=value[0,len(tokens):].tolist())
                generated.append(record)
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                stream.flush()
                if sum(p.stat().st_size for p in evidence.iterdir() if p.is_file()) > 60 * 1024**2:
                    raise ValueError('Bounded complete Java generation reports exceeded')
                print('sealed_java_generation',len(generated),row['exercise'],flush=True)
        after = state_digest(model.state_dict())
        if after != before:
            raise ValueError('Own Java policy changed during generation')
        guard.check(full=True)
        atomic_json(evidence / 'generation-seal.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
            generations_sha256=file_hash(path), questions=35, checkpoint_sha256=file_hash(checkpoint),
            before_model_state_sha256=before, after_model_state_sha256=after,
            test_feedback_seen_by_model=False, optimizer_updates=0, goal_achieved=False))
        del model, encoder
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        generation_guard = FileBindings({str(p):file_hash(p) for p in [path,evidence / 'generation-seal.json']})
        generated = list(jsonl(path))
        results = []
        for record, (row, suite, reference, prompt, tokens, budget) in zip(generated, layouts):
            generation_guard.check()
            measured = dict(record,full_file_delivery=False,compilation_passed=False,published_tests_passed=False)
            try:
                delivered = extract_files(record['raw_response'], suite['expected_solution_files'])
            except ValueError as error:
                measured['delivery_error'] = str(error)
            else:
                measured['full_file_delivery'] = True
                execution = suites.run(delivered,suite,reference['result']['inventory'])
                measured.update(execution=execution,compilation_passed=execution['compilation_passed'],
                    published_tests_passed=execution['published_tests_passed'])
            _case_pass(measured)
            results.append(measured)
            atomic_json(evidence / ('execution-' + row['exercise'] + '.json'), measured)
            print('scored_java_module',row['exercise'],measured['published_tests_passed'],flush=True)
        guard.check(full=True)
        generation_guard.check(full=True)
        suites.guard.check(full=True)
        report = dict(checkpoint_sha256=file_hash(checkpoint), data_sha256=data_sha, manifest_sha256=manifest_sha,
            source_archive_sha256=refs['source_archive_sha256'], Java_toolchain_manifest_sha256=refs['Java_toolchain_manifest_sha256'],
            test_framework_manifest_sha256=refs['test_framework_manifest_sha256'],reference_report_sha256=file_hash(ref_path),
            declared_generation_budget=2048,protocol=PROTOCOL,results=results,
            full_file_delivery_rate=sum(r['full_file_delivery'] for r in results)/35,
            compilation_success_rate=sum(r['compilation_passed'] for r in results)/35,
            published_module_pass_rate=sum(r['published_tests_passed'] for r in results)/35,
            generations_sha256=file_hash(path),generation_seal_sha256=file_hash(evidence/'generation-seal.json'),
            all_model_bytes_unchanged=True,all35_generations_sealed_before_test_execution=True,
            candidate_promoted=False,goal_achieved=False,limitations=[
                'Reused group-held publisher development modules; semantic prior exposure is not exhaustively audited.',
                'Original provided helper files are explicit model inputs; no publisher test or reference answer reaches generation.',
                'Published functional tests do not prove arbitrary Java correctness or asymptotic optimality.',
                'Same-process JUnit event inventories are not adversarial score-forgery proof.'])
        seal(report,runtime,files,data=data,digest=data_sha,device=device,tokenizer_sha256=metadata['tokenizer_sha256'])
        if sum(p.stat().st_size for p in evidence.iterdir() if p.is_file()) + len(json.dumps(report).encode()) > 64*1024**2:
            raise ValueError('Complete64MiB Java report budget exceeded')
        atomic_json(output,report)
        return report
    except Exception as error:
        atomic_json(evidence/'failure.json',dict(error=str(error),automatic_retry=False,goal_achieved=False))
        raise
    finally:
        suites.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data',default=str(ROOT/'data/exercism-support-context-v3/valid.jsonl'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:2')
    args=parser.parse_args()
    evaluate(args.checkpoint,args.data,args.output,args.device)


if __name__=='__main__':
    main()
