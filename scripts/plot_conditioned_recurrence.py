"""Export the measured own recurrence/control losses without training dependencies."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'reports/conditioned-recurrent-code-ablation-v1/comparison.json'
    source_bytes = source.read_bytes(); data = json.loads(source_bytes)
    outcomes = data['outcomes']; recurrent = outcomes['recurrent-equal-targets']
    depths = [1, 2, 4]; values = [recurrent['held_losses'][str(depth)]['nll'] for depth in depths]
    controls = [(outcomes[name]['completed_steps'], outcomes[name]['held_losses']['1']['nll'])
                for name in ('ordinary-equal-targets', 'ordinary-equal-layers')]
    assert all(row['coding']['functional_passes'] == 0 and row['coding']['tasks'] == 24 for row in outcomes.values())
    fig, ax = plt.subplots(figsize=(8.4, 5.1), constrained_layout=True)
    ax.plot(depths, values, 'o-', color='#2356A4', linewidth=2, label='Recurrent branch: 1,024 training updates')
    for (steps, value), color in zip(controls, ['#C7771A', '#218A65']):
        ax.axhline(value, color=color, linestyle='--', linewidth=1.6, label=f'Single-pass control: {steps:,} updates')
        ax.annotate(f'{value:.4f}', (4.06, value), xytext=(0, 7), textcoords='offset points',
                    color=color, fontsize=10, bbox={'facecolor':'white', 'edgecolor':'none', 'pad':1})
    for depth, value in zip(depths, values):
        ax.annotate(f'{value:.4f}', (depth, value), xytext=(0, 10), textcoords='offset points', ha='center')
    ax.set(xlabel='Inference passes through shared core (recurrent branch)',
           ylabel='Held next-token loss (NLL; lower is better)', xticks=depths,
           xlim=(.75, 4.5), ylim=(1.46, 2.01))
    ax.set_title('Extra passes help this recurrent model,\nbut both single-pass training controls perform better', loc='left', pad=18)
    ax.grid(axis='y', alpha=.18); ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    fig.supxlabel('Same 64 held windows / 32,704 targets. One seed; no uncertainty estimate.\n2,560-update control matches decoder-layer/token executions, not all FLOPs or wall time.\nAll trained branches: 0/24 functional Python/JavaScript passes (256 output tokens).',
                  fontsize=9)
    output = ROOT / 'reports/conditioned-recurrent-loss-comparison-v1'
    for suffix in ('.png', '.pdf'): fig.savefig(output.with_suffix(suffix), dpi=180)
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps({'source':str(source.relative_to(ROOT)),
        'source_sha256':hashlib.sha256(source_bytes).hexdigest(), 'recurrent_depths':depths,
        'recurrent_losses':values, 'single_pass_control_update_loss':controls,
        'matplotlib_version':matplotlib.__version__, 'scope':'Measured loss comparison only; not executable-code or frontier capability.',
        'goal_achieved':False},indent=2)+'\n')


if __name__ == '__main__': main()
