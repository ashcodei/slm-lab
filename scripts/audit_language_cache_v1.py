"""Check actual own-model cached/full logits on fixed failed-code histories."""
from contextlib import nullcontext
from datetime import datetime,timezone
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.recurrent_ablation import state_digest
from slm_lab.training import Encoder,load_checkpoint

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'reports/language-completed-cache-audit-v1'
    if output.exists():raise ValueError('Preserve earlier full-size cache diagnostic')
    device=torch.device('cuda:3');torch.cuda.set_device(device)
    if torch.cuda.mem_get_info(device)[0]<12*1024**3:
        raise ValueError('Sufficient shared-device capacity required')
    torch.cuda.set_per_process_memory_fraction(.18,device)
    torch.set_num_threads(2)
    output.mkdir()
    try:
        checkpoint=ROOT/'checkpoints/extended-curriculum-v1/language-pretraining/model.pt'
        data=ROOT/'data/codenet-algorithm-assessment-v1'
        generations=ROOT/'reports/codenet-language-completed-baseline-v1/generations.jsonl'
        seal=generations.parent/'generation-seal.json'
        binding=json.loads(seal.read_text())
        if file_hash(generations)!=binding['generations_sha256'] or file_hash(checkpoint)!=binding['checkpoint_sha256']:
            raise ValueError('Exact previously sealed own model/history required')
        files=[Path(__file__),ROOT/'src/slm_lab/model.py',ROOT/'src/slm_lab/training.py',
            ROOT/'src/slm_lab/recurrent_ablation.py',ROOT/'src/slm_lab/codenet_checkpoint_v1.py',
            checkpoint,checkpoint.parent/'tokenizer.json',data/'questions.jsonl',generations,seal]
        guard=FileBindings({str(p):file_hash(p) for p in files})
        model,metadata=load_checkpoint(checkpoint,device,training_state=False);model.eval()
        tokenizer=Tokenizer.from_file(str(checkpoint.parent/'tokenizer.json'))
        encoder=Encoder(tokenizer,model.config,data,image_size=448)
        questions=list(map(json.loads,(data/'questions.jsonl').open()))
        saved=list(map(json.loads,generations.open()))
        selected=[0,4,14,26]
        before=state_digest(model.state_dict())
        atomic_json(output/'plan.json',{'created_at':datetime.now(timezone.utc).isoformat(),
            'source_files_sha256':{str(p):file_hash(p) for p in files},'question_indices':selected,
            'sealed_history_lengths':[0,1536],'comparison_steps':[0,1,8],
            'FP32_max_abs_tolerance':.002,'BF16_max_abs_diagnostic_tolerance':.5,
            'shared_GPU_memory_fraction':.18,'new_generations_or_optimizer_updates':0,'goal_achieved':False})
        records=[]
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for precision in ['FP32','BF16']:
                for position in selected:
                    question,generation=questions[position],saved[position]
                    if question['id']!=generation['id']:raise ValueError('Exact held history order required')
                    prompt=[encoder.bos]+encoder.tokens(encoder.prompt(question))
                    body=generation['generated_token_ids']
                    for history_length in [0,1536]:
                        guard.check()
                        ids=torch.tensor([prompt+body[:history_length]],device=device)
                        amp=torch.autocast('cuda',dtype=torch.bfloat16) if precision=='BF16' else nullcontext()
                        with amp:
                            state=model.prefill(ids)
                            initial_logits=state['logits'][0,-1].float().clone()
                            initial_length=ids.shape[1]
                            for step in range(9):
                                if step in [0,1,8]:
                                    whole=model(input_ids=ids)['logits'][0].float()
                                    cached=state['logits'][0,-1].float()
                                    reference=whole[-1]
                                    error=float((reference-cached).abs().max())
                                    future_error=float((whole[initial_length-1]-initial_logits).abs().max())
                                    records.append({'id':question['id'],'precision':precision,'history_length':history_length,
                                        'decode_steps':step,'text_context_tokens':ids.shape[1],
                                        'cached_full_max_abs_error':error,'future_suffix_prefix_max_abs_error':future_error,
                                        'cached_argmax':int(cached.argmax()),'full_argmax':int(reference.argmax()),
                                        'argmax_matches':bool(cached.argmax()==reference.argmax())})
                                    del whole
                                if step<8:
                                    token=torch.tensor([[body[history_length+step]]],device=device)
                                    state=model.decode(token,state['cache'])
                                    ids=torch.cat((ids,token),dim=1)
                        del state,ids
                        print('checked_full_size_cache',precision,question['id'],history_length,flush=True)
        after=state_digest(model.state_dict())
        if before!=after:raise ValueError('Inference audit changed model state')
        guard.check(full=True)
        fp32=[r for r in records if r['precision']=='FP32'];bf16=[r for r in records if r['precision']=='BF16']
        result={'completed_at':datetime.now(timezone.utc).isoformat(),'checkpoint_sha256':binding['checkpoint_sha256'],
            'records':records,'comparisons':len(records),'FP32_max_abs_error':max(r['cached_full_max_abs_error'] for r in fp32),
            'BF16_max_abs_error':max(r['cached_full_max_abs_error'] for r in bf16),
            'FP32_future_suffix_prefix_max_abs_error':max(r['future_suffix_prefix_max_abs_error'] for r in fp32),
            'BF16_future_suffix_prefix_max_abs_error':max(r['future_suffix_prefix_max_abs_error'] for r in bf16),
            'argmax_agreements':sum(r['argmax_matches'] for r in records),
            'before_model_state_sha256':before,'after_model_state_sha256':after,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device),
            'FP32_within_declared_tolerance':all(r['cached_full_max_abs_error']<=.002 and r['future_suffix_prefix_max_abs_error']<=.002 for r in fp32),
            'BF16_within_declared_diagnostic_tolerance':all(r['cached_full_max_abs_error']<=.5 and r['future_suffix_prefix_max_abs_error']<=.5 for r in bf16),
            'optimizer_updates':0,'new_model_generations':0,'model_saved':False,'goal_achieved':False,
            'scope':'Fixed actual own706M long/short sealed histories, both precisions, cache/full-forward and future-suffix invariance. Numerical diagnostic only; not model capability or proof covering every possible input.'}
        atomic_json(output/'results.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
    except Exception as error:
        atomic_json(output/'failure.json',{'error':str(error),'optimizer_updates':0,'goal_achieved':False})
        raise


if __name__=='__main__':main()
