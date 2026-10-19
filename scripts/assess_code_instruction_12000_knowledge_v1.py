"""Use unchanged raw/native knowledge scorers on the exact own12000-update policy."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import torch

from prepare_codenet_native_contract_pack_v2 import occupied_writer_reservation
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.knowledge_evaluation import evaluate as raw_evaluate
from slm_lab.knowledge_native_evaluation_v1 import evaluate as native_evaluate

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',choices=['raw','native'],required=True)
    args=parser.parse_args();native=args.protocol=='native'
    name='code-instruction-12000-native-knowledge-v1' if native else 'code-instruction-12000-knowledge-v1'
    reports=ROOT/'reports'/name
    if reports.exists():raise ValueError('Preserve all earlier own-model assessment outputs')
    occupied=occupied_writer_reservation();reserve=26*1024**3-occupied['credited_bytes']
    if shutil.disk_usage(ROOT).free<reserve+32*1024**2:raise ValueError('Bounded32MiB assessment above remaining writer reserve required')
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(device)[0]<12*1024**3:
        raise ValueError('Sufficient shared-device BF16 capacity required before capped inference')
    torch.cuda.set_per_process_memory_fraction(.14,device);torch.set_num_threads(1)
    checkpoint=ROOT/'checkpoints/code-instruction-12000-assessment-v1/model.pt';token=checkpoint.parent/'tokenizer.json'
    extraction_path=checkpoint.parent/'extraction.json';extraction=json.loads(extraction_path.read_text())
    protocol_path=ROOT/'reports/code-instruction-12000-assessment-protocol-v1.json';protocol=json.loads(protocol_path.read_text())
    if file_hash(checkpoint)!=extraction['inference_checkpoint_sha256'] or protocol['checkpoint_sha256']!=extraction['inference_checkpoint_sha256']:
        raise ValueError('Exact own12000-update inference snapshot required')
    metadata=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    expected=extraction['expected_metadata']
    if (expected['steps']!=12000 or any(metadata.get(k)!=v for k,v in expected.items())
            or metadata['tokenizer_sha256']!=file_hash(token) or 'optimizer' in metadata):
        raise ValueError('Exact own-model metadata/tokenizer without optimizer required')
    del metadata
    data=ROOT/'data/knowledge-eval-v1/questions.jsonl';audit=data.with_suffix('.audit.json')
    rows=[json.loads(s) for s in data.open()];ids=[r['id'] for r in rows];domains=Counter(r['domain'] for r in rows)
    if len(rows)!=559 or len(set(ids))!=559:raise ValueError('Every original559 question required')
    files=[Path(__file__),checkpoint,token,extraction_path,protocol_path,data,audit,
        ROOT/'src/slm_lab/codenet_checkpoint_v1.py',ROOT/'scripts/prepare_codenet_native_contract_pack_v2.py',
        ROOT/'src/slm_lab/codenet_native_contracts_v1.py',ROOT/'src/slm_lab/knowledge_evaluation.py',
        ROOT/'src/slm_lab/knowledge_native_evaluation_v1.py']
    bound={str(p):file_hash(p) for p in files};guard=FileBindings(bound);runtime=capture()
    for p,h in protocol['unchanged_evaluation_source_files_sha256'].items():
        if file_hash(p)!=h:raise ValueError('Fixed matched evaluation source changed')
    reports.mkdir()
    def status(message,**fields):
        atomic_json(reports/'status.json',dict(status=message,updated_at=datetime.now(timezone.utc).isoformat(),
            optimizer_updates=0,candidate_promoted=False,goal_achieved=False,**fields))
    try:
        atomic_json(reports/'plan.json',dict(recorded_at=datetime.now(timezone.utc).isoformat(),checkpoint=str(checkpoint),
            checkpoint_sha256=bound[str(checkpoint)],expected_completed_metadata=expected,source_files_sha256=bound,
            runtime=runtime,questions=559,questions_by_domain=dict(domains),generation_budget=32,device=str(device),memory_fraction=.14,
            storage_reservation=dict(occupied_verified_writer=occupied,remaining_writer_reserve_bytes=reserve),
            all_fixed_questions_required=True,protocol='fixed native diagnostic; raw remains primary' if native else 'unchanged original raw protocol',
            optimizer_updates=0,candidate_promoted=False,goal_achieved=False))
        status('assessing every fixed broad-domain question on the own12000-update policy',questions=559)
        output=reports/'knowledge.json';evaluate=native_evaluate if native else raw_evaluate
        evaluate(checkpoint,data,output,str(device),max_examples=559,max_new_tokens=32)
        report=json.loads(output.read_text())
        if (report['checkpoint_sha256']!=bound[str(checkpoint)] or report['questions_sha256']!=bound[str(data)]
                or report['overall']['questions']!=559 or report['skipped']
                or [r['id'] for r in report['predictions']]!=ids or Counter(r['domain'] for r in report['predictions'])!=domains):
            raise ValueError('Exact own checkpoint and every fixed question/gold/domain required')
        guard.check(full=True)
        if capture()!=runtime:raise ValueError('Knowledge implementation/runtime changed')
        summary=dict(completed_at=datetime.now(timezone.utc).isoformat(),checkpoint_sha256=bound[str(checkpoint)],questions=559,
            generated_correct=sum(r['generated_correct'] for r in report['predictions']),
            likelihood_correct=sum(r['likelihood_correct'] for r in report['predictions']),
            overall=report['overall'],by_domain=report['by_domain'],report_sha256=file_hash(output),
            label='Own code instruction12000; matched intermediate sample with earlier-exposure qualifications, not final high-end/independent capability.',
            full_scope_is_the_prepared_sample_only=True,earlier_or_current_pretraining_overlap_fully_excluded=False,
            optimizer_updates=0,candidate_promoted=False,goal_achieved=False)
        if native:summary['native_letter_likelihood_correct']=sum(r['native_letter_likelihood_correct'] for r in report['predictions'])
        atomic_json(reports/'summary.json',summary);status('complete full-question assessment; broader capability unverified',
            questions=559,generated_correct=summary['generated_correct'],likelihood_correct=summary['likelihood_correct'])
        print(json.dumps(summary),flush=True)
    except Exception as error:
        status('knowledge assessment stopped; preserve all partial outputs and inspect',error=str(error));raise


if __name__=='__main__':main()
