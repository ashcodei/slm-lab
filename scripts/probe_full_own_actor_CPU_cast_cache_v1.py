"""Measure two actual full-size CPU training profiles with exact step recovery.

This is a resource/correctness probe, not a quality comparison or production
training. Each profile uses identical complete original public training pairs,
a frozen own984Mbase, every15Mactor parameter and a separate unused critic.
"""
from datetime import datetime,timezone
from pathlib import Path
from contextlib import nullcontext
import hashlib,json,os,random,tempfile,time,traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/full-own-actor-CPU-cast-cache-v1'
BASE=ROOT/'checkpoints/code-first-continuation-v3/standalone/model.pt'

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh GPU-hidden resource experiment required')
    import copy,numpy as np,torch
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint,capture_rng
    from slm_lab.CPU_frozen_cast_cache_v1 import CPUFrozenCastCache
    from slm_lab.own_dense_standalone_v1 import load
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference
    from slm_lab.own_dense_actor_checkpoint_v1 import make_state,restore_state
    from slm_lab.own_function_trajectory_preferences_v1 import OwnFunctionTrajectoryPreferences
    from slm_lab.own_function_trajectory_preference_stage_v1 import gradients
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    tests=ROOT/'reports/CPU-frozen-cast-cache-v1-tests.log'
    if '3 passed' not in tests.read_text():raise ValueError('Actual cached/output/gradient/invalidation controls required')
    data=OwnFunctionTrajectoryPreferences(ROOT/'data/own-function-trajectory-preferences-v1','optimization')
    profile_source=ROOT/'reports/completed-code-public-function-rollouts-CPU-v1/plan.json';source=json.loads(profile_source.read_text())
    files={**data.source_files_sha256,**{str(p):file_hash(p) for p in [BASE,profile_source,tests,ROOT/'tests/test_CPU_frozen_cast_cache_v1.py',
        *python_sources(ROOT,[Path(__file__)]) ]}};guard=FileBindings(files)
    base,_=load(BASE,expected_checkpoint_sha256=files[str(BASE)],expected_sources=source['expected_standalone_sources'])
    model=OwnDenseActor(base).eval();initial=copy.deepcopy(model.parts.state_dict());base_digest=tensor_digest(base.named_parameters())
    if model.counts()['actual_unique']!=999706442 or torch.cuda.is_initialized():raise ValueError('Actual complete own CPU under1B model required')
    batches=[data.optimization_positions[:2],data.optimization_positions[2:4]]
    if any(len(row)!=2 for row in batches):raise ValueError('Four complete admitted optimization-only source functions required')
    OUT.mkdir();settings=dict(threads=2,learning_rate=.0001,betas=[.9,.95],weight_decay=.01,gradient_clip=1.,seed=978123)
    plan=dict(owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
        source_files_sha256=files,actual_student_counts=model.counts(),profiles=['CPU_BF16_standard','CPU_BF16_cached'],training_settings=settings,
        actual_original_source_batches=batches,cpu_affinity=sorted(os.sched_getaffinity(0)),torch_version=str(torch.__version__),
        warmup_backward_without_optimizer_before_timing=True,actual_logical_updates_per_profile=2,
        no_GPU_allocation_or_production_updates=True,neural_quality_not_evaluated=True,goal_achieved=False)
    atomic_json(OUT/'plan.json',plan);plan_sha=file_hash(OUT/'plan.json');results={}
    def equal(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    for profile in plan['profiles']:
        model.parts.load_state_dict(initial);optimizer=torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],
            lr=settings['learning_rate'],betas=tuple(settings['betas']),weight_decay=settings['weight_decay'],foreach=False)
        reference=FrozenOwnActorReference(model,sources=dict(own_complete_source_sha256=files[str(BASE)]))
        amp=lambda:torch.autocast('cpu',dtype=torch.bfloat16)
        cache=CPUFrozenCastCache() if profile=='CPU_BF16_cached' else None
        if cache is not None:cache.attach(model.policy);cache.attach(reference.policy)
        sources=dict(resource_plan_sha256=plan_sha,own_complete_model_sha256=files[str(BASE)],CPU_profile_sha256=hashlib.sha256(profile.encode()).hexdigest())
        parameters=[p for _,p in model.named_selected(('actor',))];records=[];timings=[]
        random.seed(settings['seed']);np.random.seed(settings['seed']);torch.manual_seed(settings['seed'])
        gradients(model,data,batches[0],reference,method='chosen_CE',amp=amp);optimizer.zero_grad(set_to_none=True)
        def step(i,timed):
            optimizer.zero_grad(set_to_none=True);start=time.perf_counter()
            record=gradients(model,data,batches[i],reference,method='chosen_CE',amp=amp)
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,settings['gradient_clip'],error_if_nonfinite=True));optimizer.step();optimizer.zero_grad(set_to_none=True)
            elapsed=time.perf_counter()-start;record.update(step=i+1,gradient_norm_before_clip=norm);records.append(record)
            if timed:timings.append(elapsed)
            return record
        def moments():return tensor_digest((str(i)+'.'+k,v[k]) for i,v in enumerate(optimizer.state.values()) for k in ('step','exp_avg','exp_avg_sq'))
        first=step(0,True)
        with tempfile.TemporaryDirectory(prefix='slm-CPU-precision-',dir='/dev/shm') as tmp:
            path=Path(tmp)/'after1.pt'
            atomic_checkpoint(path,make_state(model,optimizer,groups=('actor',),sources=sources,completed=1,planned=2,
                cursor=dict(actual_batches=batches[:1],profile=profile),metrics=records,device='cpu'),512*1024**2,0)
            wanted=step(1,True);wanted_weights=tensor_digest(model.parts.named_parameters());wanted_adam=moments();wanted_rng=capture_rng('cpu')
            saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            count,records=restore_state(model,optimizer,saved,groups=('actor',),expected_sources=sources,planned=2,
                expected_cursor=dict(actual_batches=batches[:1],profile=profile),device='cpu')
            if (count!=1 or step(1,False)!=wanted or tensor_digest(model.parts.named_parameters())!=wanted_weights
                    or moments()!=wanted_adam or not equal(wanted_rng,capture_rng('cpu'))):
                raise ValueError('Actual CPU profile next source update, every actor/Adam value and private RNG must recover exactly')
            del saved
        if tensor_digest(base.named_parameters())!=base_digest or not torch.equal(model.parts['critic'].weight,initial['critic.weight']) or torch.cuda.is_initialized():
            raise ValueError('Every frozen dense/vision/word/critic value and GPU ownership must remain untouched')
        if cache is not None:cache.assert_current()
        results[profile]=dict(final_actor_parts_sha256=tensor_digest(model.parts.named_parameters()),final_Adam_sha256=moments(),
            cast_cache_inventory=None if cache is None else cache.inventory(),actual_update_seconds=timings,mean_actual_update_seconds=sum(timings)/len(timings),
            logical_updates=2,physical_optimizer_calls=3,actual_selected_Adam_leaves=len(optimizer.state),
            exact_next_source_update_actor_weights_Adam_and_private_RNG=True,records=records,
            actual_counts=model.counts(),neural_quality_not_evaluated=True,production_updates=0)
        atomic_json(OUT/(profile+'-complete.json'),dict(complete=True,**results[profile],goal_achieved=False))
        if cache is not None:cache.close()
        del optimizer,reference
    for key in ['records','final_actor_parts_sha256','final_Adam_sha256']:
        if results['CPU_BF16_standard'][key]!=results['CPU_BF16_cached'][key]:raise ValueError('Cached training must preserve every actual source metric, actor weight and Adam value exactly')
    guard.check(full=True)
    atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,actual_profiles=results,
        measured_standard_to_cached_BF16_update_speed_ratio=results['CPU_BF16_standard']['mean_actual_update_seconds']/results['CPU_BF16_cached']['mean_actual_update_seconds'],
        both_profiles_produce_identical_actual_metrics_actor_weights_and_Adam=True,
        first_profile_order_and_only_two_updates_not_a_general_throughput_claim=True,
        both_actual_full_size_next_update_private_state_recoveries_pass=True,
        original_GPU_and_training_jobs_unmodified=True,source_files_sha256=files,goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT/'complete.json'),standard_BF16_seconds=results['CPU_BF16_standard']['mean_actual_update_seconds'],cached_BF16_seconds=results['CPU_BF16_cached']['mean_actual_update_seconds'])),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        target=OUT/'failure.json' if OUT.exists() else ROOT/'reports/full-own-actor-CPU-cast-cache-v1-parent-failure.json'
        atomic_json(target,dict(error=repr(error),traceback=traceback.format_exc(),goal_achieved=False));raise
