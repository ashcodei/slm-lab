"""Read-only inference export from one pinned, atomically published checkpoint.

The writer may replace its pathname while this reader retains the old inode.
Only complete model values and their actual source prefix are repackaged. No
optimizer/private RNG is imported or certified here for training recovery.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import random
import stat

import numpy as np
import torch

from .corpus import file_hash
from .deterministic_positions_v1 import PROTOCOL
from .fast_joint_checkpoint_v2 import FORMAT as CORE_FORMAT
from .fast_joint_inference_v2 import FORMAT as NATIVE_FORMAT,sources as legacy_sources
from .own_dense_growth_v1 import FORMAT as DENSE_FORMAT
from .own_dense_standalone_v1 import FORMAT as STANDALONE_FORMAT
from .parallel_function_repair_stage_v1 import value_digest
from .public_pretraining_checkpoint_v1 import FORMAT as STATE_FORMAT,validate
from .public_pretraining_native_v1 import provenance,sources


def fingerprint(info):return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)


class PinnedCheckpoint:
    def __init__(self,path):
        self.original_path=Path(path).resolve(strict=True)
        if Path(path).is_symlink():raise ValueError('Only the actual regular checkpoint path may be pinned')
        self.fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW);info=os.fstat(self.fd)
        if not stat.S_ISREG(info.st_mode):os.close(self.fd);raise ValueError('A complete regular durable checkpoint is required')
        self.initial=fingerprint(info);self.path=Path('/proc/self/fd')/str(self.fd)

    def check(self):
        if fingerprint(os.fstat(self.fd))!=self.initial:raise ValueError('Pinned checkpoint inode was modified in place')

    def load(self):
        self.check();value=torch.load(self.path,map_location='cpu',weights_only=True,mmap=True);self.check();return value

    def identity(self):
        self.check();digest=file_hash(self.path);self.check()
        return dict(original_path=str(self.original_path),device=self.initial[0],inode=self.initial[1],bytes=self.initial[2],
            mtime_ns=self.initial[3],checkpoint_sha256=digest,pinned_read_only_actual_inode=True,
            pathname_may_be_atomically_replaced_without_changing_this_snapshot=True)


@contextmanager
def pin(path):
    value=PinnedCheckpoint(path)
    try:yield value;value.check()
    finally:os.close(value.fd)


def resources(free_bytes,source_bytes,memory_stat,memory_limit,*,export_maximum=5*1024**3,writer_maximum=14*1024**3,
              reserve=26*1024**3,working=32*1024**3,other_headroom=64*1024**3):
    needed=source_bytes+export_maximum+writer_maximum+reserve
    live=sum(memory_stat[k] for k in ['anon','shmem','kernel'])
    if free_bytes<needed:raise ValueError('Pinned old state, independent export, full main atomic writer and reserve must all fit')
    if live+working+other_headroom>memory_limit:raise ValueError('Complete evaluation working set and original-job headroom must fit the actual cgroup')
    return dict(actual_free_bytes=free_bytes,source_checkpoint_bytes=source_bytes,maximum_export_bytes=export_maximum,
        maximum_main_writer_bytes=writer_maximum,minimum_persistent_reserve_bytes=reserve,required_free_bytes=needed,
        actual_nonreclaimable_resident_bytes=live,evaluation_working_allowance=working,other_job_headroom=other_headroom,
        cgroup_memory_limit=memory_limit,reclaimable_file_cache_not_counted_twice=True)


def validate_source(state,plan,digest,schedule,data,journal,*,world=4,device_profile='cuda'):
    if (state.get('format')!=STATE_FORMAT or state.get('schedule_identity_sha256')!=schedule.identity_sha256
            or state.get('selected_source_budget')!=schedule.selected_budget(data) or state.get('goal_achieved') is not False):
        raise ValueError('The exact complete declared public source schedule is required')
    core=state['core'];completed=core.get('completed');native=core.get('model',{})
    if (core.get('format')!=CORE_FORMAT or core.get('mode')!='train' or core.get('run_plan_sha256')!=digest
            or core.get('training_options')!=plan['training_options'] or core.get('world_size')!=world
            or core.get('goal_achieved') is not False or type(completed) is not int or not 0<completed<plan['planned_steps']
            or native.get('format')!=NATIVE_FORMAT or native.get('position_interpolation')!=PROTOCOL
            or native.get('run_plan_sha256')!=digest or native.get('arm')!='trained'
            or native.get('completed_updates')!=completed or native.get('whole_source_epochs_completed')!=0
            or native.get('candidate_promoted') is not False or native.get('goal_achieved') is not False
            or native.get('model',{}).get('format')!=DENSE_FORMAT
            or native['model'].get('sources')!=legacy_sources(plan,digest)
            or native['model'].get('actual_counts')!=plan['actual_counts']
            or len(journal)<completed or journal[:completed]!=core.get('metrics')
            or device_profile not in ['cpu','cuda']):
        raise ValueError('The exact partial training cursor, model/source envelope and complete original journal prefix must agree')
    validate(core,plan,schedule,data)
    rngs=core.get('rngs',[])
    if len(rngs)!=world:raise ValueError('All actual producer-private RNG records must remain in the source')
    for rng in rngs:
        random.Random(0).setstate(rng['python']);n=rng['numpy']
        np.random.RandomState(0).set_state((n['kind'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
        torch.Generator(device='cpu').set_state(rng['torch'].cpu())
        cuda=rng['cuda']
        if device_profile=='cuda':
            if not isinstance(cuda,torch.Tensor) or cuda.device.type!='cpu' or cuda.dtype!=torch.uint8 or cuda.ndim!=1 or not 16<=cuda.numel()<=65536:
                raise ValueError('Every actual producer CUDA RNG byte record must be present; it is not imported on CPU')
        elif cuda is not None:raise ValueError('The declared CPU fixture cannot contain foreign CUDA RNG state')
    return dict(actual_completed_updates=completed,planned_updates=plan['planned_steps'],all_complete_source_prefix_records_verified=True,
        every_source_position_target_mask_modality_learning_rate_and_Adam_clock_reconstructed=True,
        actual_producer_world=world,all_private_producer_RNG_sha256=[value_digest(v) for v in rngs],
        optimizer_and_private_RNG_not_imported_or_certified_for_training_recovery=True,
        original_source_schedule_sha256=schedule.identity_sha256,response_EOS_targets=sum(int(r['response_EOS_targets']) for r in core['metrics']))


def inference_package(state,plan,digest):
    """Preserve all model values, retie serialization, change provenance only."""
    completed=state['core']['completed'];old=state['core']['model']['model']
    if not 0<completed<plan['planned_steps'] or old.get('sources')!=legacy_sources(plan,digest):
        raise ValueError('A separately validated real partial public-training model required')
    network=dict(old['network'])
    if not torch.equal(network['embedding.weight'],network['lm_head.weight']):raise ValueError('The original tied vocabulary values differ')
    network['lm_head.weight']=network['embedding.weight']
    dense={**old,'network':network,'sources':sources(plan,digest)}
    return dict(format=STANDALONE_FORMAT,complete_dense=dense,provenance=provenance(plan,'step-'+str(completed),completed),
        completed_updates=completed,actual_counts=old['actual_counts'],tokenizer_sha256=plan['tokenizer_sha256'],position_interpolation=PROTOCOL,
        old_checkpoint_optimizer_or_training_reference_needed_for_inference=False,candidate_promoted=False,goal_achieved=False)


def exact_values(model,package):
    dense=package['complete_dense'];model.assert_inventory()
    if model.counts()!=package['actual_counts']:raise ValueError('Every actual registered under1B value must be counted')
    for name,module in [('network',model.network),('heads',model.heads)]:
        live=module.state_dict();saved=dense[name]
        if live.keys()!=saved.keys() or any(not torch.equal(value,saved[key]) for key,value in live.items()):
            raise ValueError('Every complete cold-loaded neural tensor must be bit-exact to its pinned source')
    return dict(every_registered_network_head_and_tied_table_value_bit_exact=True,actual_counts=model.counts(),
        complete_own_inference_requires_only_packaged_model_and_tokenizer=True)
