"""Full frozen-source/native readback and fresh execution of every candidate.

This new source-only worker waits for the exact existing producer to finish.
It never changes that process or any neural worker. All source positions and
fresh replay failures/abstentions remain in the audit. Roles are assigned to
whole duplicate components before fresh outcomes; no training is admitted.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.public_code_candidate_roles_v1 import assign_roles, verify_role_separation, SEED
from slm_lab.public_code_disposition_readback_v1 import rebuild, row_sha, verify_observation
from slm_lab.public_code_instruction_screen_v1 import parse_whole, observe
from slm_lab.public_reasoning_protected_text_v1 import population, overlaps
from slm_lab.public_synthetic_code_screen_policy_v1 import body_fingerprint, supervision_scope

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/public-code-instruction-candidate-v1'
QUAL = ROOT/'reports/full-public-code-shard-qualification-CPU-v1'
PILOT = ROOT/'reports/public-code-instruction-readback-CPU-v1'
OUT = ROOT/'reports/full-public-code-shard-readback-CPU-v1'
VETO = ROOT/'reports/public-code-instruction-precision-counterexample-CPU-v1'
TOKENIZER = ROOT/'data/public-reasoning-curriculum-v1/tokenizer.json'
TESTS = ROOT/'reports/full-public-code-readback-v1-tests.log'
COUNT, RESERVE, MAX_OUTPUT = 100000, 40*1024**3, 1024**3
WORKING, OTHER_HEADROOM = 16*1024**3, 64*1024**3


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic(path, value):
    pending = path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=True)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def file_state(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def memory():
    values = {k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
    return values['anon']+values['shmem']+values['kernel'], int(Path('/sys/fs/cgroup/memory.max').read_text())


def live(owner):
    try:
        fields = Path('/proc', str(owner['pid']), 'stat').read_text().split(') ', 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[19] == owner['ticks'] and fields[0] != 'Z'


def all_lines(path):
    # Unicode paragraph/line separators inside a source string are not JSONL
    # boundaries. Only complete LF-terminated records are acceptable here.
    with path.open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                raise ValueError('A completed artifact contains a partial JSONL record')
            yield json.loads(line)


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or os.environ.get('PYTHONHASHSEED') != '0' or os.environ.get('TOKENIZERS_PARALLELISM') != 'false'
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            or '19 passed' not in TESTS.read_text()):
        raise ValueError('Fresh tested one-thread CUDA-hidden nice19 source-only reader required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    producer = read(QUAL/'plan.json')
    owner = producer['owner']
    me = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    files = dict(producer['source_files_sha256'])
    files.update({str(p):sha(p) for p in [QUAL/'plan.json', TESTS,
        *python_sources(ROOT, [Path(__file__), ROOT/'tests/test_public_code_candidate_roles_v1.py',
                             ROOT/'tests/test_public_code_disposition_readback_v1.py'])]})
    if any(sha(p) != digest for p,digest in files.items()):
        raise ValueError('Immutable producer/runtime/test bytes must match before the wait')
    states = {p:file_state(p) for p in files}
    OUT.mkdir()
    atomic(OUT/'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=me,
        exact_producer_owner=owner, source_files_sha256=files,
        waits_for_both_original_complete_receipt_and_exact_producer_terminal=True,
        no_live_output_is_source_bound=True, scheduler='SCHED_IDLE', nice=os.nice(0), CPU_threads=1, CUDA_hidden=True,
        all_original_positions=COUNT, complete_native_limit=4096, protected_input_stems=31349,
        every_original_candidate_freshly_executed_once_with_all_published_assertions=True,
        same_three_second_confined_wall_bound_as_producer=True,
        fresh_failures_and_resource_abstentions_preserved_without_rerun_or_silent_drop=True,
        role_seed=SEED, role_group_keys=['complete_normalized_input_sha256','complete_program_AST_without_docstrings_sha256'],
        transitive_components=True, roles_fixed_before_fresh_outcomes=True,
        reserved_rows_never_supplied_to_optimization=True,
        historical_or_arbitrary_semantic_independence_not_certified=True,
        source_teacher='public precomputed Qwen2.5-Coder-32B-Instruct responses and tests',
        no_external_weights_or_live_teacher_API_or_GPU_lease=True,
        minimum_free_bytes=RESERVE, maximum_output_bytes=MAX_OUTPUT, CPU_working_allowance=WORKING,
        other_process_memory_headroom=OTHER_HEADROOM, parent_address_space_ceiling=24*1024**3,
        original_main_human_RL_and_all_nine_endpoints_unchanged=True,
        scope=supervision_scope(), training_admitted=False, new_production_optimizer_updates=0, goal_achieved=False))
    waits = Counter()

    def guard():
        if os.sched_getscheduler(0) != os.SCHED_IDLE or any(file_state(p) != state for p,state in states.items()):
            raise ValueError('Idle scheduler and every immutable source byte must remain fixed')

    def checked():
        wait = wait_for_reserve(ROOT, minimum_bytes=RESERVE, guard=guard)
        waits['seconds'] += wait['waited_seconds']
        waits['failed_space_observations'] += wait['failed_space_observations']
        active, limit = memory()
        rss = next(int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
        if limit-active < OTHER_HEADROOM or rss > WORKING:
            raise ValueError('Original other-worker headroom and declared reader memory budget required')

    try:
        started = time.monotonic()
        while live(owner):
            guard()
            atomic(OUT/'progress.json', dict(owner=me, phase='waiting for exact source producer terminal',
                producer_owner=owner, producer_live=True, waited_seconds=time.monotonic()-started,
                training_admitted=False, neural_forwards=0))
            time.sleep(15)
        guard()
        if not (QUAL/'complete.json').is_file() or (QUAL/'failure.json').exists():
            raise ValueError('The exact producer exited without a successful complete source receipt')
        proof = read(QUAL/'complete.json')
        if (not proof.get('complete') or proof.get('training_admitted') is not False or proof['owner'] != owner
                or proof['actual_original_positions'] != COUNT or proof['complete_protected_input_stems'] != 31349
                or proof['source_files_sha256'] != producer['source_files_sha256']):
            raise ValueError('Exact original full100000-source unadmitted completion required')
        for path in [QUAL/'complete.json', QUAL/'all100000-original-dispositions.jsonl', QUAL/'protected-overlap-configuration.json']:
            files[str(path)] = sha(path)
            states[str(path)] = file_state(path)
        if sha(QUAL/'all100000-original-dispositions.jsonl') != proof['all_original_dispositions_sha256']:
            raise ValueError('Every terminal whole-source disposition byte must match its producer receipt')
        if any(sha(p) != digest for p,digest in files.items()):
            raise ValueError('All fixed source evidence must still match before reconstruction')
        active, limit = memory()
        if limit-active < WORKING+OTHER_HEADROOM:
            raise ValueError('Full reader memory allowance must fit alongside every original worker')
        checked()
        resource.setrlimit(resource.RLIMIT_AS, (24*1024**3, 24*1024**3))
        atomic(OUT/'terminal-source-bindings.json', dict(at=datetime.now(timezone.utc).isoformat(),
            owner=me, exact_producer_owner=owner, source_files_sha256=files,
            producer_terminal=True, training_admitted=False))
        atomic(OUT/'progress.json', dict(owner=me, phase='reconstructing all100000 original rows and complete protected scan',
            training_admitted=False))
        rows = []
        for batch in pq.ParquetFile(DATA/'train-00000-of-00050.parquet').iter_batches(batch_size=256, use_threads=False):
            rows.extend(batch.to_pylist())
        prior_rows = list(all_lines(QUAL/'all100000-original-dispositions.jsonl'))
        pilot_rows = list(all_lines(PILOT/'all1024-readback-observations.jsonl'))
        if len(rows) != COUNT or len(prior_rows) != COUNT or len(pilot_rows) != 1024:
            raise ValueError('Every whole source, disposition and exact original pilot record required')
        veto = read(VETO/'semantic-veto.json')
        if sha(veto['evidence_path']) != veto['evidence_sha256'] or row_sha(rows[veto['original_shard_position']]) != veto['original_entry_sha256']:
            raise ValueError('Original independent mathematical veto must stay exact')
        bad_ast = body_fingerprint(parse_whole(rows[veto['original_shard_position']])['program'])
        scanner = [dict(id=str(i), prompt=row['input'], independent_answer=row['output'], original_entry=dict(question=row['input']))
            for i,row in enumerate(rows) if all(isinstance(row.get(k), str) and bool(row[k]) for k in ['id','input','output'])]
        hits, settings = overlaps(scanner, population(ROOT), checked=checked)
        del scanner
        if settings != read(QUAL/'protected-overlap-configuration.json'):
            raise ValueError('The whole original two-direction protected-input scan must reconstruct')
        atomic(OUT/'protected-overlap-configuration.json', settings)
        candidates = [r for r in prior_rows if r['screen_pass_candidate']]
        roles = assign_roles(candidates)
        role_summary = verify_role_separation(candidates, roles)
        if not role_summary['train_candidates'] or not role_summary['reserved_candidates']:
            raise ValueError('Both whole original train and reserved candidate populations must be nonempty')
        with (OUT/'all-original-candidate-roles.jsonl').open('x') as stream:
            for position in sorted(roles):
                stream.write(json.dumps(dict(original_shard_position=position, **roles[position]), sort_keys=True)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        atomic(OUT/'roles-fixed-before-fresh-execution.json', dict(owner=me, role_seed=SEED,
            summary=role_summary, all_candidate_roles_sha256=sha(OUT/'all-original-candidate-roles.jsonl'),
            actual_fresh_candidate_runs_so_far=0, every_candidate_preserved=True, training_admitted=False))
        tokenizer = Tokenizer.from_file(str(TOKENIZER))
        counts, fresh_counts, role_counts, reuse_counts = Counter(), Counter(), Counter(), Counter()
        targets = contexts = fresh = tested = native_rows = overlap_rows = 0
        output = OUT/'all100000-fresh-readback-observations.jsonl'
        with output.open('x') as stream:
            for position,(row, prior) in enumerate(zip(rows, prior_rows)):
                if position % 128 == 0:
                    checked()
                result, native, parsed = rebuild(row, position, hits.get(str(position)), tokenizer, veto, bad_ast,
                    prior, pilot_rows[position] if position < 1024 else None)
                counts[result['status']] += 1
                counts['original_positions'] += 1
                native_rows += native is not None
                overlap_rows += bool(result.get('protected_overlap'))
                if 'execution_reused_from_exact_fresh_pilot_readback' in result:
                    reuse_counts['reused' if result['execution_reused_from_exact_fresh_pilot_readback'] else 'new'] += 1
                observation = dict(original_shard_position=position, source_id=row.get('id'),
                    original_entry_sha256=row_sha(row), original_disposition_rebuilt_exactly=True,
                    original_screen_status=result['status'], original_candidate=result['screen_pass_candidate'],
                    complete_native=native, source_role='excluded', training_admitted=False)
                if result['screen_pass_candidate']:
                    targets += native['response_EOS_targets']
                    contexts += native['complete_context_tokens']
                    observation.update(roles[position])
                    observation.update(input_only_group_sha256=result['input_only_group_sha256'],
                        program_AST_without_docstrings_sha256=result['program_AST_without_docstrings_sha256'])
                    # No published expectation is supplied to the confined worker.
                    replay = observe(parsed, seconds=3)
                    verify_observation(parsed, replay)
                    fresh += 1
                    tested += len(parsed['cases'])
                    fresh_counts[replay['status']] += 1
                    role = roles[position]['source_role']
                    role_counts[role+'_original_candidates'] += 1
                    role_counts[role+'_original_response_EOS_targets'] += native['response_EOS_targets']
                    role_counts[role+'_fresh_'+replay['status']] += 1
                    observation.update(fresh_whole_program_execution=replay,
                        fresh_complete_pass_weak_SFT_candidate=replay['all_published_tests_passed'],
                        independent_semantic_correctness_certified=False, use_as_RL_reward_oracle_forbidden=True)
                encoded = json.dumps(observation, ensure_ascii=True)+'\n'
                if stream.tell()+len(encoded.encode()) > MAX_OUTPUT:
                    raise ValueError('Declared complete audit storage ceiling exceeded')
                stream.write(encoded)
                if (position+1) % 128 == 0:
                    stream.flush()
                    atomic(OUT/'progress.json', dict(owner=me, phase='all-source reconstruction and fresh full-candidate execution',
                        original_dispositions_rebuilt=position+1, original_counts=dict(counts), fresh_candidate_program_runs=fresh,
                        fresh_published_assertions=tested, fresh_counts=dict(fresh_counts), fixed_role_summary=role_summary,
                        original_candidate_response_EOS_targets=targets, bounded_disk_waits=dict(waits), training_admitted=False))
            stream.flush()
            os.fsync(stream.fileno())
        if (dict(counts) != proof['actual_counts'] or fresh != len(candidates)
                or targets != proof['candidate_response_EOS_targets'] or contexts != proof['candidate_complete_context_tokens']
                or reuse_counts['new'] != proof['actual_new_program_runs']
                or reuse_counts['reused'] != proof['actual_pinned_pilot_runs_reused']
                or dict(Counter(r.get('domain') for r in rows)) != proof['source_domain_counts']):
            raise ValueError('Every original count, source role and complete native budget must reconstruct')
        verify_role_separation(candidates, roles)
        checked()
        if any(sha(p) != digest for p,digest in files.items()):
            raise ValueError('Every bound original source/runtime byte must still match at completion')
        atomic(OUT/'complete.json', dict(complete=True, integrity_readback_passed=True,
            at=datetime.now(timezone.utc).isoformat(), owner=me, source_files_sha256=files,
            actual_original_positions=COUNT, actual_original_counts=dict(counts),
            all_original_candidate_programs_freshly_executed=fresh, fresh_original_assertions_required=tested,
            actual_fresh_execution_counts=dict(fresh_counts), all_fresh_candidates_passed=fresh_counts['all_published_tests_pass'] == fresh,
            fresh_failures_and_abstentions_preserved=True, fresh_failures_do_not_change_original_roles=True,
            complete_native_rows_rebuilt=native_rows, candidate_response_EOS_targets=targets,
            candidate_complete_context_tokens=contexts, complete_protected_input_stems_rechecked=31349,
            original_protected_overlap_rows=overlap_rows, fixed_role_summary=role_summary, actual_role_counts=dict(role_counts),
            all_original_candidate_roles_sha256=sha(OUT/'all-original-candidate-roles.jsonl'),
            all_original_readback_observations_sha256=sha(output), bounded_disk_waits=dict(waits),
            source_attribution='Wasi Uddin Ahmad et al., NVIDIA, OpenCodeInstruct (2025); CC-BY-4.0',
            source_teacher='public precomputed Qwen2.5-Coder-32B-Instruct; no weights or API',
            reserved_is_new_source_reservation_not_historically_unseen_benchmark=True,
            synthetic_published_test_agreement_is_not_independent_semantic_correctness=True,
            scope=supervision_scope(), training_admitted=False, production_model_modified=False,
            new_production_optimizer_updates=0, goal_achieved=False))
        print(json.dumps(dict(complete=True, fresh_programs=fresh, fresh_counts=dict(fresh_counts),
            role_summary=role_summary, training_admitted=False)), flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json', dict(owner=me, error=repr(error), automatic_retry=False,
            training_admitted=False, new_production_optimizer_updates=0, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
