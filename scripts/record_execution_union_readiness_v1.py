"""Read back actual195/374 execution inputs, four-CPU recovery and v6 gates."""
import argparse
import ast
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import shutil
import subprocess

from tokenizers import Tokenizer
import torch

from slm_lab.code_residual_collective_checkpoint_v2 import fingerprint
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.execution_code_feedback_v1 import sequence
from slm_lab.execution_preference_v1 import fingerprint as input_fingerprint
from slm_lab.execution_production_v6 import RUNTIME_FILES
from slm_lab.recurrent_ablation import state_digest

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/execution-feedback-union-v1'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-artifacts',required=True)
    parser.add_argument('--retained-artifacts',required=True)
    args=parser.parse_args();source=Path(args.cpu_artifacts).resolve();retained=Path(args.retained_artifacts).resolve()
    if not retained.is_relative_to(ROOT/'reports') or retained.exists():raise ValueError('New owned retained artifact directory required')
    logs=[ROOT/'reports/python-execution-feedback-v2-focused-tests.log',ROOT/'reports/execution-feedback-union-v1-integration-tests.log',
        ROOT/'reports/execution-feedback-union-v1-training-tests.log']
    for log,count in zip(logs,[4,10,1]):
        if f'{count} passed' not in log.read_text() or 'FAILED' in log.read_text():raise ValueError('Completed scoped focused checks required')
    ranks=[json.loads((source/f'rank-{i}.json').read_text()) for i in range(4)]
    for i,r in enumerate(ranks):
        if (r['rank']!=i or r['actual_reference_updates']!=2 or r['actual_original_pair_reference_scores']!=374
                or r['exact_weights_Adam_per_rank_RNG_metrics_and_all_source_cursors_resume'] is not True):
            raise ValueError('Actual four-worker complete-source recovery evidence differs')
    state=torch.load(source/'step2.pt',weights_only=True,mmap=True)
    base=torch.load(source/'base.pt',weights_only=True,mmap=True)
    bundle=state['sampler_state']['execution_reference_bundle'];cache=bundle['cache']['records']
    if (state['steps']!=2 or state['planned_steps']!=60 or len(state['worker_rng_states'])!=4
            or len(state['metrics'])!=2 or len(cache)!=374
            or state['frozen_backbone_state_sha256']!=state_digest(base['model'])):
        raise ValueError('Exact original complete reference/source/frozen base state required')
    for value in state['optimizer']['state'].values():
        if int(value['step'])!=2 or not all(bool(torch.isfinite(value[k]).all()) for k in ['exp_avg','exp_avg_sq']):
            raise ValueError('Finite actual residual-only Adam step2 required')
    sampler=state['sampler_state']
    if sampler['execution_source_cursors']['positive']['examples']!=8 or sampler['execution_source_cursors']['pair']['examples']!=16:
        raise ValueError('Complete actual original auxiliary exposure cursors differ')
    token=ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json';tokenizer=Tokenizer.from_file(str(token))
    pairs=list(map(json.loads,(DATA/'pairs.jsonl').read_text().splitlines()))
    positives=list(map(json.loads,(DATA/'passing-originals.jsonl').read_text().splitlines()))
    for row in pairs:
        cached=cache[row['id']]
        for name in ['chosen','rejected']:
            value=sequence(tokenizer,row['prompt'],row[name])
            if (cached[name+'_fingerprint']!=input_fingerprint(value)
                    or cached[name+'_targets']!=sum(y!=-100 for y in value[1])
                    or not math.isfinite(cached[name+'_logp'])):raise ValueError('Every actual complete cached source/mask/score must match')
    union=json.loads((DATA/'admission.json').read_text())
    if (len(positives)!=195 or len(pairs)!=374 or union['actual_maximum_context_tokens']!={'positive':4066,'chosen':4066,'rejected':4067}
            or bundle['feedback']['manifest_sha256']!=file_hash(DATA/'manifest.json')
            or bundle['feedback']['admission_sha256']!=file_hash(DATA/'admission.json')):raise ValueError('Exact real complete union identities required')
    def function(path,name):
        return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
    for name in ['auxiliary_extremes','exercise_execution_extremes']:
        if function(ROOT/'src/slm_lab/execution_production_v5.py',name)!=function(ROOT/'src/slm_lab/execution_production_v6.py',name):
            raise ValueError('Existing tested exact complete objective/extreme math must remain unchanged')
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True).stdout.splitlines();workers=[]
    for line in apps:
        pid=int(line.split(',')[0]);p=Path('/proc')/str(pid);fields=(p/'stat').read_text().split(') ',1)[1].split()
        argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();parent=int(fields[1])
        if (fields[0]=='Z' or 'slm_lab.training pretrain' not in argv or 'systems' not in argv
                or int((Path('/proc')/str(parent)/'stat').read_text().split(') ',1)[1].split()[1])!=2754743):
            raise ValueError('Exact original live systems-code GPU worker parentage required')
        workers.append(dict(pid=pid,parent_pid=parent,start_ticks=fields[19],argv=argv,GPU_record=line))
    if len(workers)!=4:raise ValueError('Four actual original GPU workers required')
    metrics=[json.loads(x) for x in (ROOT/'reports/curriculum-disk-cleanup-resumption-v1/joint.log').read_text().splitlines() if x.startswith('{"step"')]
    shutil.copytree(source,retained,ignore=shutil.ignore_patterns('rendezvous','*.lock'))
    paths=[ROOT/'src/slm_lab'/n for n in sorted(RUNTIME_FILES)]+[Path(__file__).resolve(),
        ROOT/'tests/test_python_execution_feedback_v2.py',ROOT/'tests/test_execution_feedback_union_v1.py',
        ROOT/'tests/test_execution_production_v6.py',ROOT/'tests/test_execution_feedback_union_training_v1.py']+logs
    paths+=[DATA/'admission.json',DATA/'manifest.json',DATA/'pairs.jsonl',DATA/'passing-originals.jsonl',
        ROOT/'data/python-execution-feedback-v2/admission.json',ROOT/'reports/python-execution-feedback-v2-audit.log']
    paths+=list(p for p in retained.rglob('*') if p.is_file())
    result=dict(format='own-Java-Python-execution-training-readiness-v6',observed_at=datetime.now(timezone.utc).isoformat(),
        previous_goal_turn_classification=dict(classification='progress',reason='Prepared/audited82Python originals/1236publisher cases/153assertion pairs and common source reader;11focused checks passed while original four-L40 systems training continued.'),
        current_turn_classification=dict(classification='progress',reason='Repaired allfour original Python qualifications with distinct subtests/original helper; all86originals now pass. Preserved prior views and admitted195-program/374-pair Java/Python union. Wired real union into separate v6 capacity/production worker and tested full-source CPU reference/training/recovery.'),
        Python=dict(original_programs=86,parent_tests=1255,subtests=79,assertion_pairs=164,
            prior82positives_and153pairs_unchanged=True,all_four_original_qualifications_resolved=True,
            case_count_definition='Distinct parent tests and subtests, reported separately; not independent assertions.'),
        combined_feedback=dict(positives=195,pairs=374,languages=['Java','Python'],all_parent_views_retained_exactly=True,
            maximum_context_tokens=union['actual_maximum_context_tokens'],all374complete_reference_input_masks_reconciled=True,
            encoder_truncated_sequences=0,held_group_overlap=0),
        focused_checks=dict(native_v2=4,union_and_production_gates=10,real_union_four_worker_CPU_training=1,total=15,logs=[str(p) for p in logs]),
        CPU_training_diagnostic=dict(artifacts=str(retained),ranks=ranks,counts=state['model_counts'],
            actual_reference_updates=2,exact_replayed_update=2,planned_steps=60,complete_stage=False,
            actual_all374frozen_reference_scores_embedded_and_recovered=True,
            actual195_374execution_views_and_own32768_BPE=True,
            original_code_all30_replay_providers_and16px_images_are_fixtures=True,
            actual_full857M_GPU_capacity=False,capability_result=False),
        production=dict(module='slm_lab.execution_production_v6',actual_union_reader_wired=True,
            original_v4_final_source_base_baseline_capacity_resource_storage_and_human_feedback_gates_preserved=True,
            new_source_bound_v6_capacity_proof_required=True,existing_loss_kernel_and_extreme_objectives_unchanged=True,
            actual_full857M_capacity_completed=False,new_full_source_GPU_training_started=False),
        actual_original_training=dict(status=json.loads((ROOT/'reports/dialogue-retention-v1/status.json').read_text()),
            workers=workers,latest_metric=metrics[-1],new_execution_feedback_used=False),
        remaining_requirements=['Complete original public-code/multimodal/genuine-feedback continuation and final30-source/base coverage.',
            'Complete matched final-base coding/knowledge/image baselines and actual original v4 plus new v6 four-L40/full857M capacity/reference checks.',
            'Train the complete source populations with admitted machine feedback; retain full recovery state and run unchanged correctness/efficiency and broad text/image assessments.',
            'Establish useful multimodal, genuine-human-feedback and near-high-end capabilities; existing model measurements remain weak.'],
        new_full857M_execution_feedback_updates=0,candidate_promoted=False,new_model_capability_gain_measured=False,
        high_end_capability_established=False,goal_achieved=False,files_sha256={str(p):file_hash(p) for p in paths})
    path=ROOT/'reports/execution-feedback-union-v1-readiness.json';atomic_json(path,result)
    print(json.dumps(dict(report=str(path),Python=result['Python'],union=result['combined_feedback'],
        CPU_model_counts=state['model_counts'],focused_checks=15,actual_original_step=metrics[-1]['step'],goal_achieved=False),indent=2))


if __name__=='__main__':main()
