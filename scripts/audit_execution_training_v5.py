"""Read actual admitted programs and retained CPU integration evidence; no GPU work."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from tokenizers import Tokenizer
import torch

from slm_lab.code_residual_collective_checkpoint_v2 import fingerprint
from slm_lab.code_residual_source_contract_v4 import REQUIRED_FAMILIES
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.execution_collective_v2 import feedback_identity
from slm_lab.execution_feedback_reader_v1 import ExecutionFeedback
from slm_lab.execution_preference_v1 import validate
from slm_lab.execution_production_v5 import RUNTIME_FILES

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-artifacts',required=True)
    parser.add_argument('--retained-artifacts',required=True)
    args=parser.parse_args()
    artifacts=Path(args.cpu_artifacts).resolve()
    log=ROOT/'reports/execution-training-v5-final-tests.log'
    if '10 passed' not in log.read_text() or 'FAILED' in log.read_text():
        raise ValueError('Completed current focused integration and prelaunch checks required')
    ranks=[json.loads((artifacts/f'rank-{i}.json').read_text()) for i in range(4)]
    required=['all30_replay_families_executed_twice',
        'unequal_global_pair_and_CE_target_means_match_independent_FP64_autograd',
        'exact_Adam_RNG_full_history_reference_scores_and_all_cursors_resume',
        'reference_recovered_from_checkpoint_after_external_cache_deleted',
        'actual_production_loop_stop43_failure44_recovery43_to60_exact',
        'bounded_latest60_previous55_checkpoint_rotation_and_idempotent_export',
        'actual_combined_maximum_code_image_positive_and_complete_DPO_backward']
    for i,rank in enumerate(ranks):
        if rank['rank']!=i or rank['model_counts']['total_unique']!=8688 or any(rank.get(k) is not True for k in required):
            raise ValueError('Actual own CPU fixture evidence differs')
    checkpoint=torch.load(artifacts/'lifecycle/resume.pt',map_location='cpu',weights_only=True)
    ledger=json.loads((artifacts/'lifecycle/checkpoint-ledger.json').read_text())
    if (checkpoint['steps']!=60 or checkpoint['planned_steps']!=60 or len(checkpoint['metrics'])!=60
            or len(checkpoint['worker_rng_states'])!=4 or ledger['latest']['steps']!=60 or ledger['previous']['steps']!=55):
        raise ValueError('Actual CPU complete state/rotation differs')
    saved=checkpoint['sampler_state']
    if (len(saved['execution_reference_bundle']['cache']['records'])!=7
            or saved['execution_source_cursors']['pair']['examples']!=480
            or saved['execution_source_cursors']['positive']['examples']!=240
            or set(m['replay_family'] for m in checkpoint['metrics'])!=set(REQUIRED_FAMILIES)):
        raise ValueError('Original reference, complete cursors or replay families missing')
    tokenizer_path=ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json'
    feedback=ExecutionFeedback(ROOT/'data/execution-code-feedback-v2',Tokenizer.from_file(str(tokenizer_path)),tokenizer_path)
    identity=feedback_identity(feedback,file_hash(tokenizer_path))
    positives=[feedback.positive(i) for i in range(len(feedback.positives))]
    pairs=[feedback.example(i) for i in range(len(feedback.pairs))]
    pair_targets=validate(pairs,True)
    for ids,labels in positives:
        start=next(i for i,x in enumerate(labels) if x!=-100)
        if not (ids[0]==1 and ids[-1]==2 and len(ids)<=4096 and labels==[-100]*start+ids[start:]):
            raise ValueError('Every original positive question/program/EOS must remain whole')
    feedback.guard.check(full=True)
    def extent(sequences):
        return dict(records=len(sequences),minimum_context_tokens=min(len(x[0]) for x in sequences),
            maximum_context_tokens=max(len(x[0]) for x in sequences),
            total_context_tokens=sum(len(x[0]) for x in sequences),
            total_response_EOS_tokens=sum(sum(y!=-100 for y in x[1]) for x in sequences))
    real=dict(identity=identity,positives=extent(positives),chosen=extent([p['chosen'] for p in pairs]),
        rejected=extent([p['rejected'] for p in pairs]),all_actual_pairs_validate_for_train_only_complete_DPO=True,
        pair_target_counts_fingerprint=fingerprint(pair_targets),encoder_truncated_sequences=0,
        ordinary_original_code_and_replay_populations_unchanged=True,
        source_labels='Finite original publisher-test assertions, not human preferences.',
        full857M_auxiliary_optimizer_updates=0)
    retained=Path(args.retained_artifacts).resolve()
    if not retained.is_relative_to(ROOT/'reports') or retained==ROOT/'reports':
        raise ValueError('New owned report artifact directory required')
    if retained.exists():raise ValueError('Inspect existing retained evidence instead of overwriting')
    shutil.copytree(artifacts,retained,ignore=shutil.ignore_patterns('rendezvous','*.lock'))
    def process(pid):
        p=Path('/proc')/str(pid)
        try:
            fields=(p/'stat').read_text().split(') ',1)[1].split()
            if fields[0]=='Z':return dict(pid=pid,live=False,state='Z')
            return dict(pid=pid,live=True,state=fields[0],start_ticks=fields[19],
                parent_pid=int(fields[1]),argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
        except FileNotFoundError:return dict(pid=pid,live=False)
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True).stdout.splitlines()
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True).stdout.splitlines()
    workers=[dict(process=process(int(line.split(',')[0].strip())),gpu=line) for line in apps]
    metrics=[json.loads(line) for line in (ROOT/'reports/curriculum-disk-cleanup-resumption-v1/joint.log').read_text().splitlines() if line.startswith('{"step"')]
    pending=[ROOT/'data/vqa-real-pairs-v1/manifest.json',ROOT/'data/codenet-paired-photo-replay-v1/manifest.json',
        ROOT/'data/codenet-final-source-contract-v4/manifest.json']
    paths=[ROOT/'src/slm_lab'/n for n in sorted(RUNTIME_FILES)]+[Path(__file__).resolve(),
        ROOT/'tests/test_execution_training_v5.py',ROOT/'tests/test_execution_production_v5.py',log,
        ROOT/'data/execution-code-feedback-v2/admission.json']
    paths+=list(p for p in retained.rglob('*') if p.is_file())
    report=dict(format='own-complete-execution-training-readiness-v5',observed_at=datetime.now(timezone.utc).isoformat(),
        previous_goal_turn_classification=dict(classification='progress',reason='New complete-code/all30-source CE plus full-reference DPO kernel, independently checked global gradients and actual CPU exact-resume evidence.'),
        current_turn_classification=dict(classification='progress',reason='Connected to the actual production lifecycle, corrected total-target metrics, exercised bounded stop/failed-save/recovery/export and combined auxiliary extremes; implemented separately gated capacity/production entry point.'),
        focused_tests=dict(passed=10,log=str(log),scope='Four actual CPU workers on own8688-parameter model and explicit artificial source fixtures; structural prelaunch tests, including independently bound actual auxiliary lengths.'),
        CPU_integration=dict(original_artifacts=str(artifacts),retained_artifacts=str(retained),ranks=ranks,
            final_steps=60,latest_generation=60,previous_generation=55,
            external_cache_absence_scope='Kernel reconstruction recovers the frozen reference from the full checkpoint; production still requires its immutable bound capacity/reference evidence.'),
        actual_admitted_feedback_interface=real,
        actual_original_training=dict(GPUs=gpu,workers=workers,latest_metric=metrics[-1],
            status=json.loads((ROOT/'reports/dialogue-retention-v1/status.json').read_text()),
            new_auxiliary_feedback_used=False,scope='Original public-code and later multimodal continuation; not the new execution-feedback kernel.'),
        missing_final_source_manifests=[str(p) for p in pending if not p.exists()],
        actual_full857M_execution_capacity_completed=False,actual_new_full_source_production_started=False,
        candidate_promoted=False,high_end_capability_established=False,goal_achieved=False,
        remaining_requirements=['Complete original public-data/multimodal/genuine-human-feedback continuation.',
            'Finalize actual paired questions and all30-source full coverage bundle; match exact selected final own policy.',
            'Complete unchanged matched final-base capability baselines and original v4 capacity proof.',
            'Run actual full857M v5 reference scoring/full-objective/extreme four-L40 capacity before a production plan can be ready.',
            'Train the complete original source populations plus admitted supplemental execution feedback, retain full recovery state, evaluate unchanged coding correctness/efficiency and broad text/image tasks.',
            'Demonstrate useful multimodal, human-feedback and near-high-end capabilities; existing measured coding, knowledge and vision scores do not establish these.'],
        files_sha256={str(p):file_hash(p) for p in paths})
    target=ROOT/'reports/execution-training-v5-readiness.json'
    atomic_json(target,report)
    print(json.dumps(dict(report=str(target),real_data=real,latest_original_step=metrics[-1]['step'],
        actual_GPU_worker_count=len(workers),goal_achieved=False),indent=2))


if __name__=='__main__':main()
