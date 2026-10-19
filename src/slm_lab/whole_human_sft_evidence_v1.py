"""Whole-source SFT accounting, held teacher outputs and final raw readback."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .codenet_stage_v1 import EpochOrder
from .code_residual_checkpoint_v1 import _validate_adam
from .corpus import file_hash
from .diverse_images import atomic_json


def tensor_digest(items):
    digest=hashlib.sha256()
    for name,value in items:
        t=value.detach().cpu().contiguous()
        digest.update(json.dumps([name,str(t.dtype),list(t.shape)]).encode());digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def fixed_digest(model):
    selected={id(p) for p in model.extension_parameters('language')}
    return tensor_digest((name,p) for name,p in model.named_parameters() if id(p) not in selected)


def language_digest(model):
    return tensor_digest((name,p) for name,p in model.parts.named_parameters() if name.startswith(('language_linears.','language_norms.')))


def budget(data, options):
    order=EpochOrder(len(data),options['seed']);width=options['microbatch']*options['accumulation']
    draws=[order.at(i) for i in range(options['steps']*width)];rows=[data.rows[p]['native']['response'] for p in draws]
    counts=Counter(draws)
    if len(counts)!=len(data):raise ValueError('Every actual admitted human source must appear in a full epoch')
    return dict(source_population=len(data),source_exposures=len(draws),unique_original_positions=len(counts),
        complete_passes=len(draws)//len(data),rounding_exposures=len(draws)-len(data),
        repeated_source_positions={str(p):n for p,n in counts.items() if n>1},
        complete_context_tokens=sum(r['length'] for r in rows),response_EOS_targets=sum(r['length']-r['target_start'] for r in rows))


@torch.no_grad()
def evaluate_human(model,data,output,amp,checked=lambda:None):
    if (data.family,data.role,data.purpose)!=('dialogue','valid','evaluation') or Path(output).exists():
        raise ValueError('Fresh full original held human conversation evaluation required')
    model.eval();records=[]
    for position,row in enumerate(data.rows):
        checked();batch=data.batch([position],'response',next(model.parameters()).device)
        with amp():out=model(**batch)
        labels=batch['labels'][:,1:];mask=labels!=-100
        nll=F.cross_entropy(out['logits'][:,:-1].float().reshape(-1,model.config.vocab_size),labels.reshape(-1),
                            ignore_index=-100,reduction='none').reshape(labels.shape)[mask].cpu().tolist()
        predictions=out['logits'][:,:-1].argmax(-1)[mask].cpu().tolist();targets=labels[mask].cpu().tolist()
        if not nll or any(not math.isfinite(v) or v<0 for v in nll):
            raise ValueError('Every original whole human target requires a finite teacher token score')
        records.append(dict(source_position=position,id=row['id'],group_id=row['group_id'],language=row['language'],
            context_tokens=batch['input_ids'].shape[1],response_EOS_targets=len(targets),target_ids=targets,
            teacher_argmax_ids=predictions,per_target_nll=nll,token_nll=math.fsum(nll)/len(nll),
            teacher_correct=sum(a==b for a,b in zip(predictions,targets))))
        if (position+1)%50==0:print('whole_human_validation',position+1,'/',len(data),flush=True)
    def summary(items):
        targets=sum(r['response_EOS_targets'] for r in items)
        return dict(examples=len(items),response_EOS_targets=targets,
            token_mean_nll=math.fsum(v for r in items for v in r['per_target_nll'])/targets,
            example_mean_nll=math.fsum(r['token_nll'] for r in items)/len(items),
            teacher_argmax_accuracy=sum(r['teacher_correct'] for r in items)/targets)
    report=dict(overall=summary(records),by_language={lang:summary([r for r in records if r['language']==lang]) for lang in sorted({r['language'] for r in records})},
        results=records,whole_source_responses_and_EOS=True,response_or_prompt_truncation=False,
        source_manifest_sha256=file_hash(data.root/'manifest.json'),
        scope='Complete original held human response teacher likelihood/argmax, not free-generation or factual correctness.',goal_achieved=False)
    atomic_json(Path(output),report);return report


def reconcile(root,reports,output,data):
    root,reports,output=map(Path,(root,reports,output));plan=json.loads((reports/'plan.json').read_text())
    initial=torch.load(output/'initial/model.pt',map_location='cpu',weights_only=True,mmap=True)
    initial_parts=initial['aligned_model']['extensions'];summary={}
    for arm in plan['arms']:
        receipt=json.loads((reports/(arm+'-training-complete.json')).read_text())
        native=output/arm/'model.pt';resume=output/arm/'resume.pt'
        if file_hash(native)!=receipt['checkpoint_sha256'] or file_hash(resume)!=receipt['resume_sha256']:
            raise ValueError('Actual complete source-trained checkpoint and optimizer bytes required')
        exported=torch.load(native,map_location='cpu',weights_only=True,mmap=True)
        state=torch.load(resume,map_location='cpu',weights_only=True,mmap=True);core=state['core']
        parts=exported['aligned_model']['extensions'];names=core['optimized_names']
        if (exported['run_plan_sha256']!=file_hash(reports/'plan.json') or exported['arm']!=arm
                or core['steps']!=plan['training_options']['steps'] or len(core['metrics'])!=core['steps']
                or set(parts)!=set(initial_parts) or any(not torch.equal(t,core['model']['extensions'][n]) for n,t in parts.items())
                or any(not torch.equal(t,initial_parts[n]) for n,t in parts.items() if not n.startswith(('language_linears.','language_norms.')))):
            raise ValueError('Complete exact saved trained weights and unchanged every unused extension required')
        _validate_adam(core['optimizer'],[parts[n] for n in names],core['steps'])
        options={**plan['training_options'],'learning_rate':plan['arms'][arm]['learning_rate']};expected=budget(data,options)
        if expected!=plan['source_exposure_budget']:
            raise ValueError('Every original full-source exposure/rounding/target budget must reconstruct')
        order=EpochOrder(len(data),options['seed'])
        for step,metric in enumerate(core['metrics']):
            positions=[p for group in order.rank_groups(step,0,1,options['microbatch'],options['accumulation']) for p in group]
            target_count=sum(data.rows[p]['native']['response']['length']-data.rows[p]['native']['response']['target_start'] for p in positions)
            context=sum(data.rows[p]['native']['response']['length'] for p in positions)
            if (metric['source_positions']!=positions or metric['response_EOS_targets']!=target_count or metric['complete_context_tokens']!=context
                    or metric['step']!=step+1 or metric['learning_rate']!=options['learning_rate']):
                raise ValueError('Every actual source draw, context, target normalizer and profile must reconstruct')
        if (sum(m['response_EOS_targets'] for m in core['metrics'])!=expected['response_EOS_targets']
                or sum(m['complete_context_tokens'] for m in core['metrics'])!=expected['complete_context_tokens']):
            raise ValueError('Full actual source-token exposure totals must match')
        valid=json.loads((reports/arm/'human-validation.json').read_text())
        summary[arm]=dict(actual_updates=core['steps'],actual_Adam_states=len(core['optimizer']['state']),source_exposure_budget=expected,
            held_human=valid['overall'],trained_checkpoint_sha256=file_hash(native),every_trained_tensor_equal_complete_resume=True,
            all_unused_new_extensions_equal_initial=True,original_fixed_routes_retained=receipt['all_fixed_bytes_versions_and_original_native_routes_exact'])
    selected=json.loads((reports/'selection.json').read_text())['selected_arm'];baseline='initial'
    from .code_quality import execute
    questions={r['id']:r for r in map(json.loads,(root/'data/public-code-assessment-v1/questions.jsonl').open())}
    selected_metrics={}
    for arm in dict.fromkeys([baseline,selected]):
        destination=reports/arm
        knowledge=json.loads((destination/'knowledge.json').read_text())
        if len(knowledge['predictions'])!=559:raise ValueError('Every original broad knowledge question required')
        public=[row for shard in range(4) for row in json.loads((destination/f'public-{shard}.json').read_text())['results']]
        if len(public)!=175 or {r['id'] for r in public}!=set(questions):raise ValueError('Every original supported public code problem exactly once required')
        passes=[]
        for row in public:
            if row['execution'].get('functional_passed'):
                original=questions[row['id']]
                actual=execute(dict(language='Python',code=row['code'],function=original['function'],cases=original['cases'],stress=[]))
                if not actual.get('functional_passed'):raise ValueError('Every public candidate passing program must independently reexecute')
                passes.append(row['id'])
        local={}
        for mode in ('completion','instruction'):
            rows=json.loads((destination/('code-'+mode+'.json')).read_text())['results']
            if len(rows)!=24:raise ValueError('Every original local Python/JavaScript task required')
            local[mode]=dict(syntax=sum(r['candidate'].get('syntax_valid',False) for r in rows),functional=sum(r['candidate'].get('functional_passed',False) for r in rows),efficiency=sum(r['efficiency_stress_passed'] for r in rows))
        cpp=json.loads((destination/'C++.json').read_text())
        math_result=json.loads((destination/'math-complete.json').read_text())
        if math_result['evaluated_examples']!=1319 or not math_result['all_original_IDs_prompts_generated_tokens_numeric_answers_and_scores_reconciled']:
            raise ValueError('Every original math question, raw token delivery and numeric score must reconcile')
        selected_metrics[arm]=dict(knowledge=knowledge['overall'],public_code=dict(tasks=175,syntax=sum(r['execution'].get('syntax_valid',False) for r in public),functional=len(passes),independently_reexecuted_passing_ids=passes),
            local_code=local,CPP={k:v for k,v in cpp.items() if k not in ('results','evaluation_provenance')},
            math={k:v for k,v in math_result.items() if k not in ('results','evaluation_provenance','shard_reports_sha256')},
            OCR=json.loads((destination/'ocr.json').read_text())['overall'],objects=json.loads((destination/'objects.json').read_text())['overall'],
            vision={k:v for k,v in json.loads((destination/'vision.json').read_text()).items() if k in ('images','proper_nll','blank_nll_gain','mismatch_nll_gain','caption_retrieval')})
    report=dict(completed_full_source_human_SFT=summary,predeclared_human_validation_selection=selected,matched_initial_selected_capability=selected_metrics,
        human_SFT_not_genuine_ranked_reward_DPO_PPO=True,source_or_response_truncation=False,formal_algorithmic_complexity_not_proven=True,
        public_development_and_source_overlap_qualifications_retained=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(reports/'independent-comparison.json',report);return report
