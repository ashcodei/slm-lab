"""Close actual source controls for one explicitly small CPUgradient fixture.

No model samples, full-source qualification, training admission or GPUuse.
All attempted source/control outcomes remain; trivial hints cannot qualify.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Only CUDA-hidden nice19single-thread CPUsource mechanics qualification is allowed')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import torch
    from slm_lab.active_execution_input_bindings_v1 import input_bindings
    from slm_lab.corpus import file_hash
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    from slm_lab.native_code_execution_oracle_v2 import NativeCodeExecutionOracle, safe_observation
    from slm_lab.public_token_suffix_curriculum_v1 import view, descriptor
    from slm_lab.qualified_native_token_suffix_episode_v1 import PROOF
    root = Path(__file__).resolve().parents[1]
    out = root / 'reports/native-suffix-CPU-mechanics-qualification-v1'
    if out.exists(): raise ValueError('Preserve every earlier source qualification and attempted outcome')
    corpus = ExpandedExecutionCorpus(root)
    oracle = NativeCodeExecutionOracle(corpus)
    candidate_path = root / 'reports/native-suffix-mechanics-candidates-v2.json'
    candidates = json.loads(candidate_path.read_text())
    if candidates['complete_structural_scan_of_all_original_sources'] != 3721:
        raise ValueError('The whole real publisher-train source population must be structurally scanned before fixture selection')
    files = input_bindings(root, corpus)
    files.update(oracle.sources)
    files.update({str(p): file_hash(p) for p in [Path(__file__), candidate_path,
        root / 'src/slm_lab/qualified_native_token_suffix_episode_v1.py',
        root / 'src/slm_lab/public_token_suffix_curriculum_v1.py']})
    out.mkdir()
    chosen = None
    attempts = 0
    with (out / 'all-attempted-source-controls.jsonl').open('x') as journal:
        for candidate in candidates['candidates']:
            position = candidate['position']
            source = corpus.row(position)
            native = view(corpus.tokenizer, source['original'], *corpus.sequence(position), 4)
            if descriptor(native) != candidate['native_view'] or native['original_context_tokens'] > 512:
                raise ValueError('Every actual small fixture source and complete source/target budget must reconstruct')
            actual = []
            with oracle.checked_source_batch():
                for repeat in range(2):
                    gold = oracle.delivery(position, native['public_reference_prefix_ids'], native['original_missing_suffix_and_EOS_ids'],
                        delivered_action_budget=native['full_native_action_budget'], origin='original_reference_control')
                    prefix = oracle.delivery(position, native['public_reference_prefix_ids'], [2],
                        delivered_action_budget=native['full_native_action_budget'], origin='test_fixture')
                    try: legacy = corpus.score(position, native['public_reference_prefix_text'], stopped_on_eos=True)
                    except Exception as exc: legacy = dict(reward=None, status='preserved_source_adapter_abstention', error=repr(exc))
                    actual.append(dict(repeat=repeat, native_gold=gold, native_prefix_then_EOS=prefix,
                        legacy_prefix_then_EOS=safe_observation(legacy)))
            eligible = all(r['native_gold']['reward'] == 1. and r['native_prefix_then_EOS']['reward'] is not None
                and r['native_prefix_then_EOS']['reward'] < 1. and r['legacy_prefix_then_EOS'].get('reward') is not None
                and r['legacy_prefix_then_EOS']['reward'] < 1. for r in actual)
            observation = dict(source_position=position, source_id=source['id'], actual_controls=actual,
                native_view=descriptor(native), eligible_for_this_CPU_fixture=eligible, model_generations=0)
            raw = json.dumps(observation, ensure_ascii=True, allow_nan=False, separators=(',', ':')) + '\n'
            if len(raw.encode()) > 8 * 1024**2 or journal.tell() + len(raw.encode()) > 32 * 1024**2:
                raise ValueError('All actual source attempts must fit without truncation or omission')
            journal.write(raw)
            journal.flush()
            os.fsync(journal.fileno())
            attempts += 1
            if eligible:
                chosen = dict(format=PROOF, complete=True, at=datetime.now(timezone.utc).isoformat(), scope='CPU-mechanics',
                    complete_source_population_qualified=False, source_position=position, nominal_missing_non_EOS_tokens=4,
                    source_information=oracle.source_information(position), oracle_binding=oracle.binding(),
                    native_view=descriptor(native), actual_controls=actual, source_files_sha256=files,
                    fixed_structural_source_filter_for_small_fixture_not_production_scope=True,
                    actual_source_attempts=attempts, no_sampled_model_rejection_or_resampling=True,
                    source_training_admitted=False, production_RL_admitted=False, model_generations=0, model_updates=0,
                    full_original3721source_and14884view_and_CE_scopes_unchanged=True, goal_achieved=False)
                break
    if chosen is None: raise ValueError('No structurally selected source earned complete nontrivial native controls; preserve every attempt')
    corpus.guard.check(full=True)
    chosen['source_files_sha256'][str(out / 'all-attempted-source-controls.jsonl')] = file_hash(out / 'all-attempted-source-controls.jsonl')
    with (out / 'closed-qualified-controls.json').open('x') as stream:
        stream.write(json.dumps(chosen, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    if torch.cuda.is_initialized(): raise ValueError('Source-only qualification cannot initialize CUDA')
    print(json.dumps(dict(complete=True, source_position=chosen['source_position'], source_id=chosen['source_information']['source_id'],
        actual_attempted_source_count=attempts, actual_missing_targets=chosen['native_view']['actual_missing_response_EOS_targets'],
        full_original_native_horizon=chosen['native_view']['full_native_action_budget'], model_generations=0, production_RL_admitted=False)))


if __name__ == '__main__': main()
