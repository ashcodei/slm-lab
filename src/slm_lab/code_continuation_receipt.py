"""Verify actual complete-window target exposures before a training handoff."""


def verify_targets(report, steps, context, world=4, batch=1, accumulation=16):
    expected = world * batch * accumulation * (context - 1)
    if (report['stage'] != 'pretrain' or report['trainable'] != 'language' or report['world_size'] != world or
        report['steps'] != steps or report['planned_steps'] != steps or report['truncated_sequences'] != 0 or
        report['gradient_accumulation'] != accumulation):
        raise ValueError('Completed raw-code training differs from its declared component/budget')
    metrics = report['metrics']
    if len(metrics) != steps or [row['step'] for row in metrics] != list(range(1, steps + 1)):
        raise ValueError('Raw-code training update evidence is incomplete or repeated')
    if any(row['global_target_tokens'] != expected for row in metrics):
        raise ValueError('Raw-code training target exposures differ from its full packed-window budget')
    return {'completed_updates': steps, 'targets_per_update': expected,
            'actual_target_exposures': sum(row['global_target_tokens'] for row in metrics),
            'world_size': world, 'microbatch': batch, 'gradient_accumulation': accumulation,
            'context': context, 'truncated_source_windows': 0, 'goal_achieved': False,
            'scope': 'Actual finite training budget receipt; not trained-model correctness/capability.'}
