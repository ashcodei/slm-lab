"""Render fixed completed-main metrics with exact cohort/budget disclosures."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/completed-main-metrics-v1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('A fresh CPU-only completed metric artifact is required')
    main = ROOT / 'reports/phase-joint-v3'
    proof = ROOT / 'reports/completed-main-native-code-CPU-v2/complete.json'
    native = read(proof)
    if not native['all96original_native_programs_freshly_reexecuted']:
        raise ValueError('Every actual completed initial/final native program must be freshly checked')
    if any(frame['actual_native_verdict_differences'] for frames in native['actual_frame_outcomes'].values() for frame in frames.values()):
        raise ValueError('The exact native reexecution must reproduce every original syntax/functional verdict')
    bindings = {str(proof): sha(proof), str(Path(__file__)): sha(__file__)}
    rows = []; documents = {}
    for arm in ['initial', 'trained']:
        documents[arm] = {}
        for family in ['knowledge', 'math', 'ocr', 'objects', 'vision']:
            path = main / arm / (family + '.json'); report = read(path)
            documents[arm][family] = report; bindings[str(path)] = sha(path)
        totals = native['actual_endpoint_totals'][arm]
        if totals['tasks'] != 48 or totals['ordinary_Python_tasks'] != 24:
            raise ValueError('Both complete original code frames/languages and Python subsets are required')
        def add(metric, passed, total, scope):
            if type(passed) is not int or type(total) is not int or not 0 <= passed <= total:
                raise ValueError('Every actual count and denominator must remain explicit')
            rows.append(dict(endpoint=arm, metric=metric, passed=passed, evaluated=total,
                             fraction=passed/total, scope=scope))
        add('Code functional', totals['functional'], 48, 'Fresh native execution;12tasks x2languages x2frames;512token request;real generation ids not retained')
        add('Code syntax', totals['syntax'], 48, 'Same retained original programs under the constrained native protocol')
        add('Ordinary Python functional', totals['ordinary_Python_functional'], 24, 'Secondary stdlib module protocol;not the original math-only prompt contract')
        knowledge = documents[arm]['knowledge']; cases = knowledge['predictions']
        generated = sum(bool(r['generated_correct']) for r in cases)
        likelihood = sum(bool(r['likelihood_correct']) for r in cases)
        if (len(cases) != 56 or generated/56 != knowledge['overall']['generated_accuracy']
                or likelihood/56 != knowledge['overall']['likelihood_accuracy']):
            raise ValueError('Every stored knowledge-case count must reproduce the completed producer aggregate')
        add('Knowledge generated', generated, 56, 'Small selected development subset;32token greedy generation;source exposure not independently certified')
        add('Knowledge likelihood', likelihood, 56, 'Same questions;separate summed answer-letter likelihood;chance about11.5percent')
        math = documents[arm]['math']; cases = math['results']
        if len(cases) != 16 or math['correct']/16 != math['numeric_exact_accuracy']:
            raise ValueError('The unchanged small16question math budget must remain explicit')
        add('Math exact', math['correct'], 16, 'Small GSM8K development subset;256token greedy output;not1319question final test')
        ocr = documents[arm]['ocr']; cases = ocr['results']; passed = sum(bool(r['exact_match']) for r in cases)
        if len(cases) != 20 or passed/20 != ocr['overall']['exact_match_accuracy']:
            raise ValueError('All20actual OCR deliveries and their aggregate must remain')
        add('OCR exact', passed, 20, '20deliveries/16original images;native448RGB;case-sensitive normalized exact match')
        objects = documents[arm]['objects']; cases = objects['results']; passed = sum(bool(r['proper_exact']) for r in cases)
        if len(cases) != 15 or passed/15 != objects['overall']['correct_image_exact_accuracy']:
            raise ValueError('All15actual object deliveries and their aggregate must remain')
        add('Objects exact', passed, 15, '15deliveries/14photos;proper image;separate gray/wrong controls retained in source')
    for family in documents['initial']:
        a, b = documents['initial'][family], documents['trained'][family]
        for key in ['generation_budget', 'max_new_tokens', 'image_size', 'visual_tokens', 'data_sha256', 'questions_sha256', 'tokenizer_sha256', 'protocol']:
            if a.get(key) != b.get(key):
                raise ValueError('Every original matched cohort protocol, native budget and source must remain unchanged')
        key = 'predictions' if family == 'knowledge' else 'samples' if family == 'vision' else 'results'
        if len(a[key]) != len(b[key]):
            raise ValueError('No initial/final failure may be dropped')
        for first, second in zip(a[key], b[key]):
            for field in ['id', 'group_id', 'answer_index', 'expected', 'input_pixels_sha256', 'wrong_pixels_sha256']:
                if first.get(field) != second.get(field):
                    raise ValueError('Every actual original question/image/gold identity must match across endpoints')
    paths = sorted(main.glob('held-packed-*.json'), key=lambda p:0 if p.stem.endswith('initial') else int(p.stem.rsplit('-',1)[1]))
    losses = []; cohort = None
    for path in paths:
        d = read(path); profile = {k:d[k] for k in ['selected_original_positions','complete_causal_targets','original_manifest_sha256']}
        if cohort is None: cohort = profile
        if profile != cohort or d['evaluated_windows'] != 128 or d['complete_causal_targets'] != 524160:
            raise ValueError('Every completed retention check must use exactly the unchanged original cohort')
        step = 0 if path.stem.endswith('initial') else int(path.stem.rsplit('-',1)[1])
        losses.append(dict(completed_updates=step, token_mean_NLL=d['token_mean_NLL']))
        bindings[str(path)] = sha(path)
    if losses[-1]['completed_updates'] != 35284:
        raise ValueError('The actual completed main retention result is required')
    OUT.mkdir()
    for name, table in [('development-checks.csv', rows), ('fixed-code-retention.csv', losses)]:
        with (OUT/name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig, axes = plt.subplots(1,2,figsize=(12.5,5.4),gridspec_kw={'width_ratios':[1,1.15]})
    xs = [r['completed_updates'] for r in losses]; ys = [r['token_mean_NLL'] for r in losses]
    ax=axes[0]; ax.axvspan(0,12853,color='#2467A6',alpha=.08);ax.axvspan(12853,35284,color='#D36B20',alpha=.08)
    ax.plot(xs,ys,color='#23364D',marker='o',markersize=3,linewidth=2);ax.axvline(12853,color='#64748B',linestyle='--',linewidth=1)
    ax.set(title='Fixed code-token prediction',xlabel='Completed main updates',ylabel='Mean NLL · lower is better')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1000:.0f}k'));ax.grid(axis='y',alpha=.18)
    ax.text(.03,.93,'128 unchanged windows · 524,160 targets',transform=ax.transAxes,fontsize=8,color='#475569')
    ax.annotate(f'{ys[-1]:.4f}',xy=(xs[-1],ys[-1]),xytext=(-38,9),textcoords='offset points',fontsize=9)
    ax.set_ylim(min(ys)-.015,max(ys)+.02);ax.set_xlim(0,37000)
    ax=axes[1]; names=['Code functional','Knowledge generated','Math exact','OCR exact','Objects exact'];height=.32
    for offset,arm,color,label in [(-height/2,'initial','#2467A6','Before main'),(height/2,'trained','#D36B20','After main')]:
        selected=[next(r for r in rows if r['endpoint']==arm and r['metric']==name) for name in names]
        positions=[i+offset for i in range(len(names))]; values=[100*r['fraction'] for r in selected]
        ax.barh(positions,values,height=height,color=color,label=label)
        for y,value,row in zip(positions,values,selected):ax.text(value+.8,y,f'{row["passed"]}/{row["evaluated"]}',va='center',fontsize=8,color=color)
    ax.set_yticks(range(len(names)),names,fontsize=9);ax.invert_yaxis();ax.set_xlim(0,35);ax.set_xlabel('Cases passed (%) · higher is better')
    ax.set_title('Small matched development checks');ax.legend(frameon=False,fontsize=8,loc='lower right');ax.grid(axis='x',alpha=.18)
    fig.suptitle('SLM Lab · completed 35,284-update main run',fontsize=15,y=.98)
    fig.text(.035,.055,'Development cohorts are small and custom; these are not full independent benchmark scores. No new neural inference was run to draw this figure.',fontsize=8.5,color='#475569')
    fig.subplots_adjust(left=.075,right=.98,bottom=.18,top=.83,wspace=.48)
    for suffix in ['png','svg']:fig.savefig(OUT/('completed-main-metrics.'+suffix),dpi=170,facecolor='white')
    plt.close(fig)
    if any(sha(path)!=digest for path,digest in bindings.items()):
        raise ValueError('A completed original source report changed during rendering')
    artifact_hashes={p.name:sha(p) for p in OUT.iterdir()}
    (OUT/'complete.json').write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
        actual_development_rows=rows, actual_code_retention_points=losses,
        source_files_sha256=bindings, artifacts_sha256=artifact_hashes,
        OCR_character_error_rate={arm:documents[arm]['ocr']['overall']['character_error_rate'] for arm in documents},
        photo_retrieval_reported_aggregates={arm:documents[arm]['vision']['caption_retrieval'] for arm in documents},
        photo_full_retrieval_matrix_not_retained_or_reconstructed=True,
        small_development_probes_not_full_independent_benchmarks=True,
        broad_generated_capability_improvement_not_established=True, matplotlib=matplotlib.__version__,
        models_constructed=0,model_updates=0,goal_achieved=False),indent=2)+'\n')
    print(json.dumps(dict(figure=str(OUT/'completed-main-metrics.png'), rows=rows)),flush=True)


if __name__=='__main__':main()
