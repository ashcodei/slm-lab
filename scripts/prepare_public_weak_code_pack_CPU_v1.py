"""Pack the entire checked weak public coding shard with original sealed roles.

All source decisions remain. Only both-observation test-agreeing candidates
supply weak SFT targets. Generated tests never become an RL reward oracle.
No neural model, external weights, teacher API or GPU job is used.
"""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/public-weak-code-pack-CPU-v1'
DATA = ROOT / 'data/public-weak-code-SFT-v1'
SOURCE = ROOT / 'data/public-code-instruction-candidate-v1'
PRIOR = ROOT / 'reports/full-public-code-shard-qualification-CPU-v1'
FRESH = ROOT / 'reports/full-public-code-shard-readback-CPU-v1'
VETO = ROOT / 'reports/public-code-instruction-precision-counterexample-CPU-v1/semantic-veto.json'
RESERVE = 40 * 1024**3
MAX_PACK = 256 * 1024**2
MAX_DECISIONS = 128 * 1024**2


def jsonl(path):
    with Path(path).open('rb') as stream:
        for raw in stream:
            if not raw.endswith(b'\n'): raise ValueError('Every closed original JSONL record must retain its complete byte-LF boundary')
            yield json.loads(raw)


def main():
    if (OUT.exists() or DATA.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or os.environ.get('TOKENIZERS_PARALLELISM') != 'false'
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('A fresh CUDA-hidden nice19 single-thread bounded source packer is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    from record_visual_stage_and_first_repair_arm_v1 import atomic, read, sha
    from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.codenet_token_pack_v1 import CodePackWriter
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.public_code_candidate_roles_v1 import verify_role_separation
    from slm_lab.public_code_instruction_screen_v1 import parse_whole
    from slm_lab.public_synthetic_code_screen_policy_v1 import body_fingerprint
    from slm_lab.public_weak_code_pack_rows_v2 import build, ATTRIBUTION, REVISION
    tests = ROOT / 'reports/public-weak-code-pack-rows-v2-tests.log'
    if '8 passed' not in tests.read_text(): raise ValueError('All actual closed-source weak-only native/role controls must pass')
    proof = read(FRESH / 'complete.json')
    first = read(PRIOR / 'complete.json')
    if (not proof['complete'] or not proof['integrity_readback_passed'] or proof['training_admitted'] is not False
            or not terminal(proof['owner']) or not terminal(first['owner'])
            or proof['actual_original_positions'] != 100000 or proof['all_original_candidate_programs_freshly_executed'] != 24271
            or proof['actual_fresh_execution_counts'] != dict(all_published_tests_pass=24248,
                runtime_resource_or_confinement_abstention=13, published_test_mismatch=10)
            or proof['scope']['use_as_RL_reward_oracle_forbidden'] is not True):
        raise ValueError('Both complete exact terminal source audits and their weak-only scope are required')
    paths = [FRESH / 'complete.json', FRESH / 'all100000-fresh-readback-observations.jsonl',
        FRESH / 'all-original-candidate-roles.jsonl', FRESH / 'roles-fixed-before-fresh-execution.json',
        PRIOR / 'complete.json', PRIOR / 'all100000-original-dispositions.jsonl', VETO, tests,
        ROOT / 'data/public-reasoning-curriculum-v1/model-config.json',
        *python_sources(ROOT, [Path(__file__), ROOT / 'tests/test_public_weak_code_pack_rows_v2.py'])]
    files = dict(proof['source_files_sha256'])
    files.update({str(p): sha(p) for p in paths})
    if (sha(FRESH / 'all100000-fresh-readback-observations.jsonl') != proof['all_original_readback_observations_sha256']
            or sha(FRESH / 'all-original-candidate-roles.jsonl') != proof['all_original_candidate_roles_sha256']):
        raise ValueError('Every original closed observation and role byte must match')
    guard = FileBindings(files)
    def resource_guard():
        guard.check()
        if os.sched_getscheduler(0) != os.SCHED_IDLE: raise ValueError('All source work must remain idle')
    def reserve(): wait_for_reserve(ROOT, minimum_bytes=RESERVE + 16 * 1024**2, guard=resource_guard)
    reserve()
    OUT.mkdir()
    DATA.mkdir()
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    atomic(OUT / 'plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        original_source_positions=100000, original_candidate_positions=24271, selected_weak_SFT_train_rows=21739,
        original_reserved_candidates=2509, fresh_mismatch_or_abstention_rows_retained=23,
        roles_fixed_before_fresh_outcomes_and_never_reassigned=True,
        dataset='nvidia/OpenCodeInstruct', revision=REVISION, attribution=ATTRIBUTION, license='CC-BY-4.0',
        response_and_test_origin='Public precomputed Qwen2.5-Coder-32B-Instruct',
        source_text_is_synthetic_not_human_or_own_policy=True, generated_test_agreement_not_independent_semantics=True,
        use_as_RL_reward_oracle_forbidden=True, minimum_other_writer_free_bytes=RESERVE,
        maximum_pack_bytes=MAX_PACK, maximum_all_source_decision_bytes=MAX_DECISIONS,
        new_neural_models=0, new_program_executions=0, GPUleases=0, production_updates=0,
        training_admitted=False, independent_complete_pack_readback_still_required=True, goal_achieved=False))
    writer = None
    try:
        prior_rows = list(jsonl(PRIOR / 'all100000-original-dispositions.jsonl'))
        observations = list(jsonl(FRESH / 'all100000-fresh-readback-observations.jsonl'))
        roles = {r['original_shard_position']: r for r in jsonl(FRESH / 'all-original-candidate-roles.jsonl')}
        if len(prior_rows) != 100000 or len(observations) != 100000 or len(roles) != 24271:
            raise ValueError('Every original disposition, fresh observation and preassigned role is required')
        summary = verify_role_separation([r for r in prior_rows if r['screen_pass_candidate']],
            {p: {k: v for k, v in r.items() if k != 'original_shard_position'} for p, r in roles.items()})
        if summary != proof['fixed_role_summary']: raise ValueError('Every original whole duplicate component and fixed role must reconstruct')
        tokenizer = Tokenizer.from_file(str(ROOT / 'data/public-reasoning-curriculum-v1/tokenizer.json'))
        veto = read(VETO)
        copies = [(SOURCE / 'publisher-README.md', 'publisher-README.md'),
            (ROOT / 'data/public-reasoning-curriculum-v1/tokenizer.json', 'tokenizer.json'),
            (ROOT / 'data/public-reasoning-curriculum-v1/model-config.json', 'model-config.json')]
        for source, name in copies: shutil.copyfile(source, DATA / name)
        (DATA / 'DATASET-LICENSE.txt').write_text(
            ATTRIBUTION + '\nDataset: nvidia/OpenCodeInstruct\nRevision: ' + REVISION +
            '\nPublisher-declared license: CC-BY-4.0\nhttps://creativecommons.org/licenses/by/4.0/\n' +
            'Source: https://huggingface.co/datasets/nvidia/OpenCodeInstruct/tree/' + REVISION + '\n' +
            'Changes: select source-bound test-agreeing rows; retain complete original responses; add native token framing and fixed source roles.\n' +
            'Public synthetic responses and tests: Qwen2.5-Coder-32B-Instruct. No model weights or teacher API used.\n')
        class WaitingWriter(CodePackWriter):
            def guard(self):
                reserve()
                super().guard()
        writer = WaitingWriter(DATA, maximum_bytes=MAX_PACK, minimum_free_gib=40)
        counts = Counter()
        position = 0
        first_batch = next(pq.ParquetFile(SOURCE / 'train-00000-of-00050.parquet').iter_batches(
            batch_size=veto['original_shard_position'] + 1, use_threads=False)).to_pylist()
        bad_AST = body_fingerprint(parse_whole(first_batch[veto['original_shard_position']])['program'])
        del first_batch
        with (OUT / 'all100000-pack-dispositions.jsonl').open('x') as decisions:
            for batch in pq.ParquetFile(SOURCE / 'train-00000-of-00050.parquet').iter_batches(batch_size=256, use_threads=False):
                for original in batch.to_pylist():
                    if position % 128 == 0: reserve()
                    value, status = build(original, prior_rows[position], observations[position], roles.get(position), tokenizer,
                        veto_sha256=veto['original_entry_sha256'], veto_AST_sha256=bad_AST)
                    packed_position = None
                    if value is not None:
                        row, ids, labels = value
                        packed_position = len(writer.entries)
                        writer.add(row, ids, labels)
                        counts[row['split'] + '_examples'] += 1
                        counts[row['split'] + '_response_EOS_targets'] += sum(t != -100 for t in labels)
                    counts[status] += 1
                    counts['all_original_positions'] += 1
                    raw = json.dumps(dict(original_shard_position=position, source_id=original.get('id'),
                        original_fixed_role=observations[position]['source_role'], pack_disposition=status,
                        packed_position=packed_position, weak_SFT_only=value is not None,
                        generated_tests_not_RL_oracle=True), ensure_ascii=True, separators=(',', ':')) + '\n'
                    if decisions.tell() + len(raw.encode()) > MAX_DECISIONS: raise ValueError('Preserve the complete decision inventory within its declared quota')
                    decisions.write(raw)
                    position += 1
                    if position % 1000 == 0:
                        decisions.flush()
                        atomic(OUT / 'progress.json', dict(owner=owner, phase='pack complete original weak SFT rows and preserve every disposition',
                            actual_original_positions=position, actual_counts=dict(counts), training_admitted=False, new_neural_updates=0))
            decisions.flush()
            os.fsync(decisions.fileno())
        if position != 100000 or counts['train_examples'] != 21739 or counts['validation_examples'] != 2509:
            raise ValueError('Every original source position and exact preassigned positive/held population must remain')
        manifest = writer.finish(dict(source_files_sha256=files, dataset='nvidia/OpenCodeInstruct', revision=REVISION,
            attribution=ATTRIBUTION, license='CC-BY-4.0', synthetic_teacher_data=True,
            response_and_test_origin='Public precomputed Qwen2.5-Coder-32B-Instruct',
            weak_SFT_only=True, use_as_RL_reward_oracle_forbidden=True, human_preferences=False, own_model_on_policy=False,
            original_fixed_role_summary=summary, all_original_decisions_sha256=sha(OUT / 'all100000-pack-dispositions.jsonl'),
            historical_exposure_or_semantic_independence_not_certified=True, independent_complete_pack_readback_required=True))
        guard.check(full=True)
        atomic(OUT / 'complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, dataset_manifest_sha256=sha(DATA / 'manifest.json'), actual_counts=dict(counts),
            actual_pack_counts=manifest['counts'], all_original_decisions_sha256=sha(OUT / 'all100000-pack-dispositions.jsonl'),
            complete_original_source_positions=100000, complete_native_positive_rows=24248,
            actual_fixed_role_summary=summary, generated_tests_not_independent_semantic_proof=True,
            use_as_RL_reward_oracle_forbidden=True, new_program_executions=0, new_neural_models=0,
            production_updates=0, training_admitted=False, independent_complete_pack_readback_still_required=True, goal_achieved=False))
        print(str(OUT / 'complete.json'), flush=True)
    except BaseException as exc:
        if writer is not None and not writer.tokens.closed: writer.close()
        atomic(OUT / 'failure.json', dict(owner=owner, error=repr(exc), automatic_retry=False,
            every_original_and_partial_source_artifact_retained=True, training_admitted=False, production_updates=0, goal_achieved=False))
        raise


if __name__ == '__main__': main()
