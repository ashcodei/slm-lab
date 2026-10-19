"""Full fixed-horizon allocation, separate from human-reward PPO rollouts."""
import torch

from .codenet_checkpoint_v1 import capture_rng
from .dense_ppo_cached_v2 import SAMPLER
from .dense_ppo_v1 import action_outputs, prompt_source
from .native_human_reward_v1 import signature
from .native_parallel_ppo_v2 import ParallelNativePPO
from .parallel_function_repair_stage_v1 import value_digest


def stress(stage, *, checked=lambda: None):
    if type(stage) is not ParallelNativePPO: raise ValueError('The actual four-rank shared-native PPO stage is required')
    kernel = stage.kernel; model = kernel.model; stage.check(full=True)
    full = kernel.full and kernel.device.type == 'cuda'
    if full and (torch.cuda.get_device_name(kernel.device) != 'NVIDIA L40' or kernel.device.index != stage.rank):
        raise ValueError('Every full-size actual PPO replica must prove capacity on its own original L40')
    positions = sorted(range(len(kernel.data)), key=lambda p:(-kernel.data.rows[p]['native']['chosen']['target_start'],p))[:2]
    before = signature(model.parameters()); rng = value_digest(capture_rng(kernel.device))
    original_Adam = value_digest(kernel.optimizer.state_dict()); rows = []
    if full: torch.cuda.reset_peak_memory_stats(kernel.device)
    try:
        for position in positions:
            checked(); source = prompt_source(kernel.data, position); prompt = source['prompt_ids']
            horizon = kernel.options['max_new_tokens']; vocab = model.config.vocab_size
            actions = [4+i%(vocab-4) for i in range(horizon)]
            kernel.optimizer.zero_grad(set_to_none=True); model.eval()
            with torch.autocast(device_type=kernel.device.type, enabled=False):
                with torch.no_grad():
                    cache = model.policy.prefill(torch.tensor([prompt], device=kernel.device)); sampled = []
                    for step, action in enumerate(actions):
                        sampled.append(float(cache['logits'][0,-1].float().log_softmax(-1)[action]))
                        if step+1 < horizon: cache = model.policy.decode(torch.tensor([[action]],device=kernel.device),cache['cache'])
                    del cache
                    ids = torch.tensor([prompt+actions], device=kernel.device)
                    causal, _ = action_outputs(model, ids, len(prompt), kernel.amp)
                    reference, _ = action_outputs(model.base, ids, len(prompt), kernel.amp)
                    score = float(kernel.reward.score(torch.tensor([prompt+actions+[2]],device=kernel.device)))
                    actual = torch.tensor(sampled, device=kernel.device)
                    if not torch.allclose(actual,causal,atol=SAMPLER['probability_atol'],rtol=SAMPLER['probability_rtol']) or not bool(torch.isfinite(reference).all()):
                        raise ValueError('The entire fixed horizon must preserve actual cached/full causal probabilities')
                model.train(); model.policy.gradient_checkpointing = True
                logps, values = action_outputs(model, ids, len(prompt), kernel.amp)
                loss = -logps.mean()+(values-1).square().mean(); loss.backward(inputs=kernel.parameters)
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in kernel.parameters):
                    raise ValueError('The complete true maximum prompt/action shape must allocate finite real actor and critic gradients')
            rows.append(dict(source=source, actual_fixed_nonEOS_actions=horizon, actual_policy_context_tokens=len(prompt)+horizon,
                actual_reward_context_tokens=len(prompt)+horizon+1, actual_prefills=1,actual_decodes=horizon-1,
                maximum_cache_full_logp_difference=float((actual-causal).abs().max()), actual_shared_reward_score=score,
                synthetic_capacity_loss=float(loss.detach()), fixed_actions_are_not_training_or_human_feedback=True))
            kernel.optimizer.zero_grad(set_to_none=True); stage.check()
        checked()
        if (signature(model.parameters()) != before or value_digest(kernel.optimizer.state_dict()) != original_Adam
                or value_digest(capture_rng(kernel.device)) != rng or stage.completed != 0):
            raise ValueError('Capacity may allocate but never update the model/Adam/source cursor or private generators')
        return dict(complete=True,rank=stage.rank,world_size=4,actual_native_values=model.base.counts()['total_unique'],
            actual_policy_and_shared_reward_values=model.counts()['actual_unique'],maximum_two_whole_original_prompts=rows,
            actual_peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(kernel.device) if full else 0,
            actual_peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(kernel.device) if full else 0,
            actual_device_type=kernel.device.type,actual_device_index=kernel.device.index,
            full_size_original_L40_horizon_capacity_passed=full, all_parameters_Adam_and_RNG_unchanged=True,
            no_optimizer_updates=True,production_PPO_admitted=False,goal_achieved=False)
    finally: kernel.optimizer.zero_grad(set_to_none=True)
