"""Prepare full-source CE/EOS comparison only after exact native data proof.

No model load, optimizer, code execution, GPU lease or training admission.
The later owned launcher still needs a fixed completed own initializer,
actual full-size GPU capacity/readback, and all original quality assessments.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'reports/public-reasoned-code-train-data-CPU-v1'
OUT = ROOT/'reports/public-reasoned-code-stopping-plan-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def live(owner):
    try:
        fields = (Path('/proc')/str(owner['pid'])/'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:
        return False
    return fields[19] == str(owner['birth']) and fields[0] not in ('Z','X')


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))):
        raise ValueError('Fresh CUDA-hidden nice19 one-thread source-only plan owner required')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0))
    predecessor = json.loads((SOURCE/'owner-plan.json').read_text())
    logs = [ROOT/'reports'/name for name in ['public-reasoned-code-stopping-SFT-v1-tests.log',
        'public-reasoned-code-capacity-cases-v1-tests.log','public-reasoned-code-controllers-v1-tests.log']]
    for path, expected in zip(logs,['9 passed','6 passed','1 passed']):
        if expected not in path.read_text(): raise ValueError('Actual objective/stage/source/controller tests must finish first')
    modules = ['complete_response_stopping_loss_v1','parallel_public_reasoned_code_SFT_stage_v1',
        'public_reasoned_code_capacity_cases_v1','public_reasoned_code_capacity_controller_v1',
        'public_reasoned_code_training_controller_v1','public_reasoned_code_train_data_v1']
    paths = [Path(__file__).resolve(),SOURCE/'owner-plan.json',*logs,
        *[ROOT/'src/slm_lab'/(name+'.py') for name in modules],
        *[ROOT/'tests'/('test_'+name+'.py') for name in modules[:3]],
        ROOT/'tests/test_public_reasoned_code_controllers_v1.py',
        ROOT/'tests/test_public_reasoned_code_train_data_v1.py']
    files = {str(path):sha(path) for path in paths}
    owner = dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    OUT.mkdir()
    def emit(name,value):
        with (OUT/name).open('x') as stream:
            stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
    scope = dict(new_model_loads=0,new_optimizer_calls=0,new_program_executions=0,GPUcontexts=0,
        fixed_completed_initializer_not_yet_bound=True,actual_full_GPU_capacity_and_owned_launcher_still_required=True,
        original_full_code_broad_math_human_photo_word_assessments_still_required=True,
        production_training_admitted=False,quality_gain_established=False,candidate_promoted=False,goal_achieved=False)
    emit('owner-plan.json',dict(owner=owner,at=datetime.now(timezone.utc).isoformat(),exact_predecessor=predecessor['owner'],
        source_files_sha256=files,CPU_threads=1,nice=19,scheduler='SCHED_IDLE',**scope))
    try:
        while live(predecessor['owner']):
            print(json.dumps(dict(phase='waiting_exact_complete_native_train_batch_proof',predecessor=predecessor['owner'])),flush=True)
            time.sleep(30)
        if (SOURCE/'failure.json').exists() or not (SOURCE/'complete.json').is_file():
            raise ValueError('Exact native tensor proof lacks terminal success; never restart it')
        result = json.loads((SOURCE/'complete.json').read_text())
        if result.get('complete') is not True or result.get('owner') != predecessor['owner']:
            raise ValueError('Exact complete original native proof owner required')
        if shutil.disk_usage(ROOT).free < 48*1024**3:
            raise ValueError('Preserve48GiB persistent other-writer reserve')
        stat = dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
        if int(Path('/sys/fs/cgroup/memory.max').read_text())-sum(int(stat[k]) for k in ('anon','shmem','kernel')) < 52*1024**3:
            raise ValueError('4GiB own plus48GiB other-job nonfile memory reserve required')
        import torch
        from slm_lab.complete_response_stopping_loss_v1 import PROFILES,metadata
        from slm_lab.public_reasoned_code_capacity_cases_v1 import choose
        from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()
        data = PublicReasonedCodeSFTData(ROOT,views='both')
        files.update(data.source_files_sha256)
        files[str(SOURCE/'complete.json')] = sha(SOURCE/'complete.json')
        entries = data.complete_schedule(epochs=1,width=16,seed=990231)
        plan = choose(data,epochs=1,seed=990231)
        if not plan['all_four_replicas_receive_each_selected_source_in_declared_plan']:
            raise ValueError('Declared full source-extreme capacity must assign selected sources to all four ranks')
        emit('complete-source-capacity-plan.json',plan)
        totals = {profile:dict(targets=0,EOS=0,weighted_targets=0.) for profile in PROFILES}
        with (OUT/'every-original-global-objective-batch.jsonl').open('x') as stream:
            for index,positions in enumerate(entries):
                batch = data.batch(positions)
                target_count = int(batch['labels'][:,1:].ne(-100).sum())
                eos_count = int(batch['labels'][:,1:].eq(2).sum())
                assert eos_count == len(positions)
                profile_rows = {}
                for profile in PROFILES:
                    meta = metadata([data.information(p) for p in positions],profile)
                    assert meta['response_EOS_targets'] == target_count and meta['genuine_EOS_targets'] == eos_count
                    assert meta['actual_global_weighted_targets'] == target_count+(meta['EOS_weight']-1)*eos_count
                    totals[profile]['targets'] += target_count
                    totals[profile]['EOS'] += eos_count
                    totals[profile]['weighted_targets'] += meta['actual_global_weighted_targets']
                    profile_rows[profile] = meta
                stream.write(json.dumps(dict(global_step=index+1,exact_original_positions=positions,
                    actual_rank_positions=[positions[r::4] for r in range(4)],profile_metadata=profile_rows),allow_nan=False)+'\n')
        budget = data.budget(entries)
        for profile,factor in PROFILES.items():
            assert totals[profile] == dict(targets=budget['response_EOS_targets'],EOS=len(data),
                weighted_targets=budget['response_EOS_targets']+(factor-1)*len(data))
        data.audit_provenance()
        assert all(sha(path)==value for path,value in files.items()) and not torch.cuda.is_initialized()
        outputs = {str(OUT/name):sha(OUT/name) for name in
            ['complete-source-capacity-plan.json','every-original-global-objective-batch.jsonl']}
        emit('complete.json',dict(owner=owner,at=datetime.now(timezone.utc).isoformat(),complete=True,
            source_files_sha256=files,output_files_sha256=outputs,complete_original_budget=budget,
            actual_complete_profiles=totals,complete_schedule_identity=data.identity(entries),
            all_original_train_code_reasoning_and_EOS_targets_preserved=True,original_tail_not_rounded=True,
            common_complete_source_schedule_for_both_profiles=True,reserved_optimization_forbidden=True,
            public_synthetic_data_not_human_RL_feedback=True,**scope))
    except BaseException:
        emit('failure.json',dict(owner=owner,traceback=traceback.format_exc(),**scope))
        raise


if __name__ == '__main__': main()
