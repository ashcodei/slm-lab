"""Assess the code-instruction4000 intermediate snapshot on every fixed broad-domain QA item."""
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
from slm_lab.knowledge_evaluation import evaluate
from prepare_codenet_native_contract_pack_v2 import occupied_writer_reservation

ROOT=Path(__file__).resolve().parents[1]


def main():
    reports=ROOT/'reports/code-instruction-4000-knowledge-v1'
    if reports.exists():raise ValueError('Preserve every earlier own-model QA assessment')
    occupied=occupied_writer_reservation()
    remaining_writer_reserve=26*1024**3-occupied['credited_bytes']
    if shutil.disk_usage(ROOT).free<remaining_writer_reserve+32*1024**2:
        raise ValueError('Reserve32MiB above remaining persistent writer capacity')
    device=torch.device('cuda:3');torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(device)[0]<12*1024**3:
        raise ValueError('Sufficient free BF16 CUDA capacity required for the capped assessment')
    torch.cuda.set_per_process_memory_fraction(.14,device)
    reports.mkdir()
    def status(message,**fields):
        atomic_json(reports/'status.json',{'status':message,'updated_at':datetime.now(timezone.utc).isoformat(),
            'optimizer_updates':0,'candidate_promoted':False,'goal_achieved':False,**fields})
    try:
        checkpoint=ROOT/'checkpoints/code-instruction-4000-assessment-v1/model.pt'
        extraction_path=checkpoint.parent/'extraction.json'
        extraction=json.loads(extraction_path.read_text())
        if extraction['inference_checkpoint_sha256']!=file_hash(checkpoint):raise ValueError('Exact extracted4000-update inference state required')
        token=checkpoint.parent/'tokenizer.json'
        data=ROOT/'data/knowledge-eval-v1/questions.jsonl'
        audit=data.with_suffix('.audit.json')
        plan_path=ROOT/'reports/extended-curriculum-v1/plan.json'
        plan=json.loads(plan_path.read_text())
        spec=next(s for s in plan['stages'] if s['name']=='multilingual-code-instruction')
        metadata=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        expected={'stage':'sft','steps':4000,'planned_steps':spec['steps'],
                  'world_size':4,'data_sha256':spec['data_sha256']}
        if any(metadata.get(k)!=v for k,v in expected.items()):
            raise ValueError('Exact own code-instruction4000 snapshot required')
        if metadata['tokenizer_sha256']!=file_hash(token):raise ValueError('Own tokenizer differs')
        del metadata
        rows=list(map(json.loads,data.open()));ids=[row['id'] for row in rows]
        if len(ids)!=len(set(ids)):raise ValueError('Unique complete fixed question set required')
        domains=Counter(row['domain'] for row in rows)
        files=[Path(__file__),checkpoint,extraction_path,token,data,audit,plan_path,ROOT/'src/slm_lab/codenet_checkpoint_v1.py',ROOT/'scripts/prepare_codenet_native_contract_pack_v2.py',ROOT/'src/slm_lab/codenet_native_contracts_v1.py']
        bound={str(p):file_hash(p) for p in files};guard=FileBindings(bound)
        runtime=capture()
        atomic_json(reports/'plan.json',{'recorded_at':datetime.now(timezone.utc).isoformat(),
            'checkpoint':str(checkpoint),'checkpoint_sha256':bound[str(checkpoint)],
            'expected_completed_metadata':expected,'source_files_sha256':bound,'runtime':runtime,
            'questions':len(rows),'questions_by_domain':dict(domains),'generation_budget':32,'device':str(device),
            'memory_fraction':.14,'storage_reservation':{'occupied_verified_writer':occupied,'remaining_writer_reserve_bytes':remaining_writer_reserve},'all_fixed_questions_required':True,'optimizer_updates':0,
            'label':'Own code-instruction update4000; same unchanged raw knowledge protocol as completed-language baseline; not the final CodeNet starting checkpoint.',
            'contamination_scope':'The retained question audit excludes earlier exact-text matches. It does not certify the subsequently expanded foundation corpus or all semantic/clone exposure.',
            'goal_achieved':False})
        status('assessing all fixed broad-domain questions on own code-instruction4000 checkpoint',questions=len(rows))
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
            'overall':report['overall'],'by_domain':report['by_domain'],'report_sha256':file_hash(output),
            'label':'Own code-instruction update4000. Matched raw-protocol intermediate assessment; not the final new-CodeNet-stage baseline.',
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
