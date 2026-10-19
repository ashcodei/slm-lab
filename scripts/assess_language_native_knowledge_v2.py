"""Native-format and candidate-tokenization diagnostic on the same completed model and all559 QA items."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.knowledge_native_evaluation_v1 import evaluate

ROOT=Path(__file__).resolve().parents[1]


def main():
    reports=ROOT/'reports/language-completed-native-knowledge-v2'
    if reports.exists():raise ValueError('Preserve every earlier own-model QA assessment')
    # The fixed writer reservation predates the current stage's retained snapshot.
    # Credit only that verified active writer state, without deleting or changing it.
    frozen=json.loads((ROOT/'reports/extended-curriculum-v1/plan.json').read_text())
    current=json.loads((ROOT/'reports/extended-curriculum-v1/status.json').read_text())
    credit=0;credited_snapshot=None
    if current.get('stage')=='multilingual-code-instruction' and current.get('status')=='training extended public-data curriculum':
        spec=next(s for s in frozen['stages'] if s['name']==current['stage'])
        directory=ROOT/'checkpoints/extended-curriculum-v1'/current['stage']
        snapshots=sorted(directory.glob('step-*.pt'))
        if snapshots:
            candidate=snapshots[-1]
            state=torch.load(candidate,map_location='cpu',weights_only=True,mmap=True)
            if (state.get('stage')!='sft' or state.get('planned_steps')!=spec['steps']
                or state.get('world_size')!=4 or state.get('data_sha256')!=spec['data_sha256']
                or not 0<state.get('steps',0)<spec['steps'] or 'optimizer' not in state):
                raise ValueError('Only the exact live stage writer snapshot can be credited')
            credit=min(candidate.stat().st_size,9*1024**3)
            credited_snapshot={'path':str(candidate),'bytes':candidate.stat().st_size,
                'completed_steps':state['steps'],'planned_steps':state['planned_steps'],'data_sha256':state['data_sha256']}
            del state
    required_free=26*1024**3-credit+32*1024**2
    if shutil.disk_usage(ROOT).free<required_free:
        raise ValueError('Reserve remaining persistent-writer capacity plus32MiB assessment output')
    device=torch.device('cuda:1');torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(device)[0]<12*1024**3:
        raise ValueError('Sufficient free BF16 CUDA capacity required for the capped assessment')
    torch.cuda.set_per_process_memory_fraction(.18,device)
    reports.mkdir()
    def status(message,**fields):
        atomic_json(reports/'status.json',{'status':message,'updated_at':datetime.now(timezone.utc).isoformat(),
            'optimizer_updates':0,'candidate_promoted':False,'goal_achieved':False,**fields})
    try:
        checkpoint=ROOT/'checkpoints/extended-curriculum-v1/language-pretraining/model.pt'
        token=checkpoint.parent/'tokenizer.json'
        data=ROOT/'data/knowledge-eval-v1/questions.jsonl'
        audit=data.with_suffix('.audit.json')
        plan_path=ROOT/'reports/extended-curriculum-v1/plan.json'
        plan=json.loads(plan_path.read_text())
        spec=next(s for s in plan['stages'] if s['name']=='language-pretraining')
        metadata=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        expected={'stage':'pretrain','steps':spec['steps'],'planned_steps':spec['steps'],
                  'world_size':4,'data_sha256':spec['data_sha256']}
        if any(metadata.get(k)!=v for k,v in expected.items()):
            raise ValueError('Exact completed own language-stage checkpoint required')
        if metadata['tokenizer_sha256']!=file_hash(token):raise ValueError('Own tokenizer differs')
        del metadata
        rows=list(map(json.loads,data.open()));ids=[row['id'] for row in rows]
        if len(ids)!=len(set(ids)):raise ValueError('Unique complete fixed question set required')
        domains=Counter(row['domain'] for row in rows)
        files=[Path(__file__),checkpoint,token,data,audit,plan_path,ROOT/'src/slm_lab/codenet_checkpoint_v1.py',ROOT/'src/slm_lab/knowledge_native_evaluation_v1.py']
        bound={str(p):file_hash(p) for p in files};guard=FileBindings(bound)
        runtime=capture()
        atomic_json(reports/'plan.json',{'recorded_at':datetime.now(timezone.utc).isoformat(),
            'checkpoint':str(checkpoint),'checkpoint_sha256':bound[str(checkpoint)],
            'expected_completed_metadata':expected,'source_files_sha256':bound,'runtime':runtime,
            'questions':len(rows),'questions_by_domain':dict(domains),'generation_budget':32,'device':str(device),
            'memory_fraction':.18,'all_fixed_questions_required':True,'optimizer_updates':0,
            'storage_reservation':{'original_writer_reserve_bytes':26*1024**3,'verified_already_retained_writer_bytes':credit,
                'remaining_initial_free_bytes_required':required_free,'credited_active_snapshot':credited_snapshot,
                'files_deleted_or_modified':False},
            'label':'Fixed native-format diagnostic against the preserved raw-completion baseline; same own language-stage checkpoint.',
            'contamination_scope':'The retained question audit excludes earlier exact-text matches. It does not certify the subsequently expanded foundation corpus or all semantic/clone exposure.',
            'goal_achieved':False})
        status('assessing all fixed broad-domain questions on completed own language checkpoint',questions=len(rows))
        output=reports/'knowledge.json'
        evaluate(checkpoint,data,output,str(device),max_examples=len(rows),max_new_tokens=32)
        report=json.loads(output.read_text())
        if (report['checkpoint_sha256']!=bound[str(checkpoint)] or report['questions_sha256']!=bound[str(data)]
            or report['overall']['questions']!=len(rows) or report['skipped']
            or [r['id'] for r in report['predictions']]!=ids
            or Counter(r['domain'] for r in report['predictions'])!=domains):
            raise ValueError('Exact own checkpoint and every original QA item must be retained')
        guard.check(full=True)
        if capture()!=runtime:raise ValueError('Knowledge assessment implementation/runtime changed')
        summary={'completed_at':datetime.now(timezone.utc).isoformat(),'checkpoint_sha256':bound[str(checkpoint)],
            'questions':len(rows),'generated_correct':sum(r['generated_correct'] for r in report['predictions']),
            'likelihood_correct':sum(r['likelihood_correct'] for r in report['predictions']),
            'native_letter_likelihood_correct':sum(r['native_letter_likelihood_correct'] for r in report['predictions']),
            'overall':report['overall'],'by_domain':report['by_domain'],'report_sha256':file_hash(output),
            'label':'Native prompt/tokenization diagnostic, fixed after the aggregate raw baseline; no question-specific prompt edits or changed weights. No weights changed; raw baseline remains primary.',
            'full_scope_is_the_prepared_sample_only':True,'earlier_or_current_pretraining_overlap_fully_excluded':False,
            'optimizer_updates':0,'candidate_promoted':False,'goal_achieved':False}
        atomic_json(reports/'summary.json',summary)
        status('complete broad-domain stage baseline finished; broader capability unverified',
               questions=len(rows),generated_correct=summary['generated_correct'],likelihood_correct=summary['likelihood_correct'])
        print(json.dumps(summary),flush=True)
    except Exception as error:
        status('broad-domain baseline stopped; inspect preserved evidence',error=str(error))
        raise


if __name__=='__main__':main()
