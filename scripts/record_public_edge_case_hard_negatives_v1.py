"""Full-scope review of actual finite-edge-case machine preference admission."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.function_repair_GPU_contract_v1 import ADMISSION, QUEUE, handoff_admitted


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/public-edge-case-hard-negatives-v1-review'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('Fresh CUDA-hidden full-scope review required')
    mined_path = ROOT / 'reports/public-edge-case-hard-negatives-CPU-v1/complete.json'
    admitted_path = ROOT / 'reports/public-edge-case-hard-negatives-v1-readback-admission/complete.json'
    mined, admitted = read(mined_path), read(admitted_path)
    if (not mined.get('complete') or not admitted.get('complete')
            or mined['candidate_pairs'] != admitted['actual_admitted_machine_pairs']
            or admitted['actual_original_trials_reconstructed'] != 536
            or mined['original_sources'] != 57):
        raise ValueError('Complete fixed-budget mining and actual fresh whole-program admission required')
    audit = read(ROOT / 'reports/completion-audit.json')
    previous = Path(audit['latest_full_scope_review'])
    files = dict(admitted['source_files_sha256'])
    for path in [Path(__file__), mined_path, admitted_path, previous]:
        files[str(path)] = file_hash(path)
    guard = FileBindings(files)
    queue = read(ROOT / QUEUE)
    main_plan = read(ROOT / 'reports/public-pretraining-recovery-v1/plan.json')
    cpu_plan = read(ROOT / 'reports/full-function-repair-comparison-CPU-v1/plan.json')
    evaluation = read(ROOT / 'reports/public-pretraining-interim-CPU-v2/evaluation-plan.json')
    if (not handoff_admitted(ROOT, queue['owner']) or terminal(main_plan['owner'])
            or terminal(cpu_plan['owner']) or terminal(evaluation['owner'])
            or (ROOT / 'reports/public-pretraining-interim-CPU-v2/failure.json').exists()):
        raise ValueError('Original full neural runs, matched evaluation and all ten successor owners must remain live')
    raw = (ROOT / 'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes()
    prefix = raw[:raw.rfind(b'\n') + 1]
    rows = list(map(json.loads, prefix.splitlines()))
    if [r['step'] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('Complete contiguous actual public training prefix required')
    current = dict(owner=main_plan['owner'], complete_actual_updates=len(rows), planned_updates=4447,
                   response_EOS_targets=sum(int(r['response_EOS_targets']) for r in rows),
                   complete_prefix_sha256=hashlib.sha256(prefix).hexdigest())
    cpu = read(ROOT / 'reports/full-function-repair-comparison-CPU-v1/progress.json')
    status = read(ROOT / 'reports/public-pretraining-interim-CPU-v2/progress.json')
    pairs = [json.loads(v) for v in (ROOT / 'data/public-edge-case-hard-negatives-admitted-v1/pairs.jsonl').read_text().splitlines()]
    at = datetime.now(timezone.utc).isoformat()
    OUT.mkdir()
    proof = dict(complete=True, at=at, source_files_sha256=files, previous_goal_turn_classification='progress: actual partial model export, cold native controls and matched assessment launched',
                 current_goal_turn_classification='progress: source-preserving semantic negatives missed by published examples mined and freshly admitted',
                 original_positive_sources=57, lexical_candidates=536, actual_mining_program_runs=mined['counts']['actual_program_runs'],
                 actual_admitted_machine_pairs=len(pairs), distinct_positive_sources=admitted['distinct_positive_sources'],
                 actual_fresh_readback_program_runs=admitted['actual_fresh_confined_program_runs'],
                 native_response_EOS_targets=admitted['native_response_EOS_targets'],
                 original_main=current, original_CPU_comparison=cpu, matched_interim_assessment=status,
                 actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used',
                                                         '--format=csv,noheader,nounits'], text=True),
                 original_all_domain_multimodal_human_RL_and_final_benchmark_scope_preserved=True,
                 actual_new_neural_updates=0, new_generated_quality_gain_established=False, candidate_promoted=False, goal_achieved=False)
    atomic_json(OUT / 'complete.json', proof)
    lines = ['# Verified finite-edge-case coding preferences', '',
             f'The fixed miner retains all **57** previously qualified original source functions and examines all **536** declared single-token candidates. '
             f'It performs **{mined["counts"]["actual_program_runs"]}** actual confined program runs. Fresh readback admits **{len(pairs)}** machine preference pairs '
             f'from **{admitted["distinct_positive_sources"]}** original functions after **{admitted["actual_fresh_confined_program_runs"]}** additional actual program runs.', '',
             'Both programs pass every original published example. The unchanged original also passes its entire fixed finite contract; '
             'each rejected derivative fails at least one completely observed finite case. Only one lexical token differs: comments, documentation, '
             'function signatures and all other source bytes remain unchanged. Whole positive and negative responses, original prompts, masks and EOS are retained.', '',
             'These are execution-labelled source derivatives, not model rollouts, human judgments or independent benchmark examples. '
             'Passing finite checks does not prove general correctness. Unknown or resource-incomplete cases produce no wrong-code label. '
             'The reader uses fresh confined processes with the same bound numerical contracts; it is not an independent oracle implementation.', '',
             '| Contract | Single edit | Counterexample | Expected | Actual | Published examples passed |',
             '|---|---|---|---|---|---:|']
    for row in pairs[:10]:
        witness = row['counterexample']; patch = row['mutation']['patch']
        lines.append('| ' + row['contract_family'] + ' | `' + patch['old'] + ' → ' + patch['new'] + '` | `' + repr(witness['case']['args']) +
                     '` | `' + repr(witness['case']['expected']) + '` | `' + repr(witness['raw']['exception_only'] or witness['raw']['stdout'].strip()) +
                     '` | ' + str(row['published_examples_passed']) + ' |')
    lines += ['', f'The admitted pairs retain {admitted["native_response_EOS_targets"]["positive"]:,} positive and '
              f'{admitted["native_response_EOS_targets"]["negative"]:,} negative response/EOS targets. No model has consumed this new data.', '',
              f'The unchanged four-L40 run is at {len(rows):,}/4,447 updates in this review. The original CPU/GPU repair studies, '
              'full human preference and execution-RL pipelines, all 17 pretraining families and all full final benchmark populations remain unchanged. '
              'The step 2,048 matched CPU assessment is active. **No new model accuracy gain is established; all ten final requirements remain incomplete.**', '']
    (OUT / 'metrics.md').write_text('\n'.join(lines))
    review = copy.deepcopy(read(previous))
    message = (f' All 57 qualified original sources now have 536 fixed source-preserving lexical mutation trials. '
               f'{len(pairs)} machine preference pairs from {admitted["distinct_positive_sources"]} functions escape every original public example '
               f'and fail an observed finite contract. All trials/native pairs reconstruct; fresh admission performs {admitted["actual_fresh_confined_program_runs"]} actual '
               'program runs. Whole original positives, task prompts, complete negatives and genuine EOS remain. '
               'This adds unconsumed execution-labelled training data; it is not human feedback, on-policy RL, a model-quality gain or a correctness proof.')
    for key in ['public_data', 'human_feedback', 'high_end_llm_capability', 'any_code_correctness_and_efficiency']:
        review['requirements'][key]['evidence'] += message
    review['requirements']['serious_full_model_training']['evidence'] += f' Latest complete main prefix is {len(rows)}/4447; new data-mining neural updates remain 0.'
    if any(v['status'] != 'incomplete_for_requested_final_model' for v in review['requirements'].values()):
        raise ValueError('Source preference admission does not complete a final-model requirement')
    review.update(at=at, previous_full_scope_review=str(previous), current_goal_turn_classification=proof['current_goal_turn_classification'],
                  latest_actual_edge_case_preference_admission=proof,
                  source_files_sha256={**files, str(OUT / 'complete.json'): file_hash(OUT / 'complete.json')}, goal_achieved=False)
    atomic_json(OUT / 'review.json', review)
    guard.check(full=True)
    maps = [main_plan['source_files_sha256'], cpu_plan['source_files_sha256'], queue['source_files_sha256'], evaluation['source_files_sha256']]
    maps += [read(ROOT / item['queue'])['source_files_sha256'] for item in read(ROOT / ADMISSION)['new_controllers']]
    for name in ['reports/completion-audit.json', 'README.md', 'RESEARCH.md']:
        if any(str(ROOT / name) in mapping for mapping in maps):
            raise ValueError('Never change live source-bound documents')
    audit.update(at=at, latest_full_scope_review=str(OUT / 'review.json'), requirements=review['requirements'],
                 latest_actual_main_runtime=current, latest_edge_case_preference_admission=str(OUT / 'complete.json'), goal_achieved=False)
    atomic_json(ROOT / 'reports/completion-audit.json', audit)
    paragraph = (f'Finite-edge-case coding preferences, **{at}**: all 57 qualified source functions and 536 single-token trials are checked; '
                 f'fresh readback admits **{len(pairs)} machine preference pairs** from {admitted["distinct_positive_sources"]} functions '
                 f'after {admitted["actual_fresh_confined_program_runs"]} new confined program runs. Both sides pass every published example; '
                 'the derivative fails a complete finite contract. Original positives/prompts and all other source bytes remain unchanged. '
                 f'The four-L40 main continues at **{len(rows)}/4,447**; matched step 2,048 evaluation remains active. '
                 '**The new pairs have not been trained; no accuracy gain is established.** All full human/RL/final-benchmark stages remain. '
                 '**All ten final requirements remain incomplete.** [Data evidence](reports/public-edge-case-hard-negatives-v1-review/metrics.md), '
                 '[full review](reports/public-edge-case-hard-negatives-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        path = ROOT / name; first, rest = path.read_text().split('\n', 1)
        path.write_text(first + '\n\n' + paragraph + rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT / 'review.json'), admitted_pairs=len(pairs), new_neural_updates=0, goal_achieved=False)), flush=True)


if __name__ == '__main__':
    main()
