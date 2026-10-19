"""Read actual four-rank selected states into a disposable CPU export actor.

This is whole-state source reconstruction, not a four-rank training substitute.
Every complete trajectory and Adam value is checked before any selected copy.
CUDA generators remain hashed bytes; CPU export never replays their semantics.
"""
import torch

from .codenet_checkpoint_v1 import capture_rng
from .code_residual_checkpoint_v1 import _validate_adam, _validate_rng
from .corpus import file_hash
from .native_parallel_ppo_checkpoint_v2 import FORMAT, common_digest, prefix
from .native_parallel_ppo_v2 import ParallelNativePPO, parallel_binding, FORMAT as PARALLEL_FORMAT
from .native_shared_ppo_v1 import NativeSharedPPOStage
from .own_dense_actor_checkpoint_v1 import selected
from .parallel_function_repair_stage_v1 import value_digest


class ReadOnlyRecipe:
    """Reuse only pure source/trajectory math; there is no pretend collective."""
    positions = ParallelNativePPO.positions
    cursor = ParallelNativePPO.cursor
    validate_rows = ParallelNativePPO.validate_rows
    validate_record = ParallelNativePPO.validate_record

    def __init__(self,kernel,rank):
        self.kernel,self.rank=kernel,rank;self.world=4
        self.options,self.binding=parallel_binding(kernel);self.binding_sha256=value_digest(self.binding)
        self.history_binding={**self.binding,'physical_rank':rank}
        self.initial_global_record_sha256=value_digest(dict(format=PARALLEL_FORMAT,binding_sha256=self.binding_sha256))

    def check(self): self.kernel.check()


def materialize(kernel, artifacts, *, expected_completed, checked=lambda:None):
    if (type(kernel) is not NativeSharedPPOStage or kernel.device.type!='cpu' or len(artifacts)!=4
            or kernel.optimizer.state or kernel.completed!=0 or kernel.history is not None):
        raise ValueError('One fresh disposable CPU native/actor and all four actual private source artifacts are required')
    rng=value_digest(capture_rng('cpu'));wanted_common=None;generators=[];first=None;readbacks=[]
    for rank,spec in enumerate(artifacts):
        checked();kernel.check(full=True)
        if file_hash(spec['path'])!=spec['sha256']:raise ValueError('Every exact original private checkpoint byte must match')
        state=torch.load(spec['path'],map_location='cpu',weights_only=True,mmap=True);recipe=ReadOnlyRecipe(kernel,rank)
        if (state.get('format')!=FORMAT or state.get('rank')!=rank or state.get('world_size')!=4
                or state.get('completed')!=expected_completed or state.get('binding')!=recipe.binding
                or state.get('binding_sha256')!=recipe.binding_sha256 or state.get('cursor')!=recipe.cursor(expected_completed)
                or state.get('common_sha256')!=spec['common_sha256'] or common_digest(state)!=spec['common_sha256']
                or state.get('private_RNG_sha256')!=spec['private_RNG_sha256']
                or value_digest(state.get('private_RNG'))!=spec['private_RNG_sha256']
                or state.get('no_resident_trajectory_history') is not True
                or any(state.get(k) is not False for k in ['candidate_promoted','goal_achieved','useful_own_output_alignment_proved'])):
            raise ValueError('All four actual common states, original complete source cursors and private generators must agree')
        kernel.model.validate_export(state['policy'],expected_sources=kernel.sources)
        _validate_adam(state['optimizer'],kernel.parameters,expected_completed)
        if state['optimizer']['param_groups']!=kernel.optimizer.state_dict()['param_groups']:
            raise ValueError('Every original ordinary selected AdamW option must match')
        private=state['private_RNG'];_validate_rng(dict(private,cuda=None),'cpu')
        if kernel.full and (not isinstance(private['cuda'],torch.Tensor) or private['cuda'].dtype!=torch.uint8
                or private['cuda'].ndim!=1 or private['cuda'].numel()<1):
            raise ValueError('Every actual full GPUprivate generator must retain complete original bytes')
        if wanted_common is not None and wanted_common!=state['common_sha256']:
            raise ValueError('Every actual rank must retain the exact same learned actor/critic and Adam values')
        wanted_common=state['common_sha256'];generators.append(state['private_RNG_sha256'])
        result=prefix(recipe,state['history'],state['history_budgets'],state['last_global_record_sha256'],checked)
        if result['original_prompt_draws']!=recipe.cursor(expected_completed)['actual_prompt_draws']:
            raise ValueError('The entire original source/action history must reconcile before selected reconstruction')
        readbacks.append(result)
        if rank==0:first=state
    if len(set(generators))!=4:raise ValueError('All four independently retained actual private generators must remain distinct')
    if value_digest(capture_rng('cpu'))!=rng:raise ValueError('Read-only actual source reconstruction must preserve the CPU generator')
    checked();kernel.check(full=True)
    named=selected(kernel.model,kernel.optimizer,('actor','critic'))
    with torch.no_grad():
        for name,p in named:p.copy_(first['policy']['parts'][name])
    if value_digest(capture_rng('cpu'))!=rng:raise ValueError('No private training generator may be restored by CPU reconstruction')
    return dict(complete=True,expected_completed=expected_completed,actual_world=4,
        actual_unique_policy_values=kernel.model.counts()['actual_unique'],common_sha256=wanted_common,
        every_actual_private_state_and_original_trajectory_checked=True,all_four_private_generator_identities=generators,
        actual_complete_prefixes=readbacks,CUDA_private_generator_semantics_not_replayed=True,
        CPU_RNG_unchanged=True,production_GPU_admitted=False,goal_achieved=False)
