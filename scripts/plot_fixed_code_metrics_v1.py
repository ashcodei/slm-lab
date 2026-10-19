"""Render completed fixed-cohort diagnostics without constructing a model."""
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/fixed-code-metrics-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only fixed metric artifact required')
    paths=sorted((ROOT/'reports/phase-joint-v3').glob('held-packed-*.json'),key=lambda p:0 if p.stem.endswith('initial') else int(p.stem.rsplit('-',1)[1]))
    if len(paths)<17:raise ValueError('Every already completed fixed metric point required')
    rows=[];protocol=None;bindings={}
    for path in paths:
        row=json.loads(path.read_text());step=0 if path.stem.endswith('initial') else int(path.stem.rsplit('-',1)[1])
        current={k:row[k] for k in ['original_split','selected_original_positions','original_valid_population','evaluated_windows',
            'complete_causal_targets','maximum_original_context','original_manifest_sha256','full_original_valid_stream']}
        if protocol is None:protocol=current
        if (current!=protocol or row['evaluated_windows']!=128 or row['complete_causal_targets']!=524160
                or row['no_optimizer_or_gradient'] is not True or not math.isfinite(row['token_mean_NLL'])):
            raise ValueError('All original positions, complete budgets, source hashes and measurement scopes must match')
        rows.append(dict(completed_updates=step,token_mean_NLL=row['token_mean_NLL'],perplexity=math.exp(row['token_mean_NLL']),
            source_report=str(path.relative_to(ROOT)),source_report_sha256=sha(path)));bindings[str(path)]=sha(path)
    reader_path=ROOT/'reports/main-matched-code-complete-v1-readback/complete.json';reader=json.loads(reader_path.read_text())
    totals=reader['actual_matched_endpoint_totals'];endpoints=['foundation12853','instruction14336']
    if any(totals[n]['tasks']!=48 for n in endpoints):raise ValueError('Both unchanged original native task populations required')
    native=[dict(endpoint=name,completed_updates=step,tasks=48,syntax=totals[name]['syntax'],functional=totals[name]['functional'],
        empirical_efficiency=totals[name]['producer_efficiency']) for name,step in zip(endpoints,[12853,14336])]
    bindings[str(reader_path)]=sha(reader_path);bindings[str(Path(__file__).resolve())]=sha(__file__)
    OUT.mkdir()
    with (OUT/'fixed-packed-code.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (OUT/'fixed-native-code.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(native[0]));writer.writeheader();writer.writerows(native)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1.2,1]})
    blue,orange='#2467A6','#D36B20';xs=[r['completed_updates'] for r in rows];ys=[r['token_mean_NLL'] for r in rows]
    ax=axes[0];ax.axvspan(0,12853,color=blue,alpha=.07);ax.axvspan(12853,max(xs),color=orange,alpha=.07)
    ax.plot(xs,ys,color='#23364D',linewidth=2,marker='o',markersize=3)
    ax.axvline(12853,color='#64748B',linestyle='--',linewidth=1)
    ax.annotate('Instruction + vision starts',xy=(12853,min(ys)),xytext=(16100,min(ys)+.025),fontsize=8,
        arrowprops={'arrowstyle':'-','color':'#64748B'})
    ax.annotate(f'{ys[-1]:.4f}',xy=(xs[-1],ys[-1]),xytext=(-38,10),textcoords='offset points',fontsize=9)
    ax.set(title='Fixed packed-code token prediction',xlabel='Completed main updates',ylabel='Mean NLL per target token · lower is better')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1000:.0f}k'))
    ax.set_xlim(0,max(xs)*1.035);ax.set_ylim(min(ys)-.015,max(ys)+.018);ax.grid(axis='y',alpha=.18)
    ax.text(.03,.93,'128 unchanged windows · 524,160 targets',transform=ax.transAxes,fontsize=8,color='#475569')
    ax=axes[1];categories=['Syntax valid','Functionally correct','Efficient'];width=.34
    for i,(row,color,label) in enumerate(zip(native,[blue,orange],['Foundation 12,853','Instruction 14,336'])):
        offset=-width/2 if i==0 else width/2;positions=[j+offset for j in range(3)]
        values=[row['syntax'],row['functional'],row['empirical_efficiency']]
        ax.bar(positions,values,width=width,color=color,label=label)
        for position,value in zip(positions,values):ax.text(position,value+1,f'{value}/48',ha='center',va='bottom',fontsize=8,color=color)
    ax.set_xticks(range(3),categories,fontsize=9);ax.set_ylim(0,56);ax.set_yticks([0,12,24,36,48]);ax.grid(axis='y',alpha=.18)
    ax.set(title='Actual generated-code outcomes',ylabel='Tasks passed');ax.legend(frameon=False,fontsize=8,loc='upper right',bbox_to_anchor=(1,.83))
    ax.text(.03,.91,'Two fixed prompt frames · 512-token budget',transform=ax.transAxes,fontsize=8,color='#475569')
    fig.suptitle('SLM Lab · code retention and correctness',fontsize=15,x=.5,y=.98)
    fig.text(.03,.035,'Token prediction and generated-program correctness are separate measurements. Native scores use the two older checkpoints shown.',fontsize=9,color='#475569')
    fig.subplots_adjust(left=.08,right=.98,bottom=.18,top=.84,wspace=.3)
    fig.savefig(OUT/'code-metrics.png',dpi=170,facecolor='white');fig.savefig(OUT/'code-metrics.svg',facecolor='white');plt.close(fig)
    if any(sha(path)!=expected for path,expected in bindings.items()):raise ValueError('An original completed metric changed during rendering')
    report=dict(at=datetime.now(timezone.utc).isoformat(),complete=True,original_protocol=protocol,actual_completed_metric_points=len(rows),
        last_main_token_loss_update=max(xs),last_token_mean_NLL=ys[-1],native_checkpoint_updates=[12853,14336],actual_native_outcomes=native,
        source_files_sha256=bindings,artifacts_sha256={p.name:sha(p) for p in sorted(OUT.iterdir())},
        rendering_environment=dict(python=sys.executable,matplotlib=matplotlib.__version__),
        token_loss_not_a_native_correctness_or_all_domain_score=True,no_new_neural_inference_or_training=True,
        models_constructed=0,model_updates=0,goal_achieved=False)
    (OUT/'complete.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(figure=str(OUT/'code-metrics.png'),metric_points=len(rows),latest_token_loss_update=max(xs),native_functional=[r['functional'] for r in native])),flush=True)


if __name__=='__main__':main()
