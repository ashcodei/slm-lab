"""Check every complete held QA prompt against the actual own context budget.

No answers are generated. MMLU, GSM8K and NQ keep their existing native prompt
definitions and full fixed generation budgets; overlength inputs fail closed.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/full-held-native-context-preflight-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only native held input preflight required')
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.held_question_registry_v2 import original_question
    from slm_lab.knowledge_chat_diagnostic_v1 import question_prompt
    from slm_lab.math_corpus import prompt as math_prompt
    from slm_lab.model import ModelConfig
    from slm_lab.training import Encoder
    torch.set_num_threads(1)
    proof_path=ROOT/'reports/phase-joint-step2048-readback-v1/complete.json';proof=json.loads(proof_path.read_text())
    path=ROOT/'checkpoints/phase-joint-v3/step-2048/model.pt'
    if not proof['complete'] or file_hash(path)!=proof['source_files_sha256'][str(path)]:raise ValueError('Exact already read-back own native architecture required')
    raw=torch.load(path,map_location='cpu',weights_only=True,mmap=True);config=ModelConfig(**raw['model']['config']);del raw
    token_path=ROOT/'data/tokenizer-corpus/tokenizer.json';tokenizer=Tokenizer.from_file(str(token_path))
    phase_path=ROOT/'reports/phase-joint-v3/plan.json';phase=json.loads(phase_path.read_text())
    if file_hash(token_path)!=phase['tokenizer_sha256'] or tokenizer.token_to_id('<bos>')!=1:raise ValueError('One exact own native tokenizer required')
    pool=ROOT/'data/full-held-question-pool-v2';manifest=json.loads((pool/'manifest.json').read_text())
    mmlu=pool/'mmlu-pro.original-test.jsonl';math=ROOT/'data/math-reasoning-v2/test.jsonl';nq=ROOT/'data/nq-human-questions-v1/valid.jsonl'
    if file_hash(mmlu)!=manifest['files_sha256'][mmlu.name]:raise ValueError('Whole original MMLU question pool required')
    for source in [math,nq]:
        if file_hash(source)!=manifest['source_files_sha256'][str(source)]:raise ValueError('Original held source bytes changed')
    encoder=Encoder(tokenizer,config,ROOT/'data/nq-human-questions-v1',448)
    records=[];summary={}
    for benchmark,source,count,budget in [('MMLU-Pro',mmlu,12032,32),('GSM8K',math,1319,512),('NQ-open',nq,3610,32)]:
        rows=[json.loads(s) for s in source.open()]
        if len(rows)!=count or len({str(r['id']) for r in rows})!=count:raise ValueError('Every distinct original held row required')
        lengths=[]
        for row in rows:
            if benchmark=='MMLU-Pro':text=question_prompt(row,'chat')
            elif benchmark=='GSM8K':text='User: '+math_prompt(original_question(benchmark,row))+'\nAssistant:'
            else:
                if row['prompt']!=original_question(benchmark,row)+'\nAnswer briefly.':raise ValueError('Whole unchanged original NQ stem required')
                text=encoder.prompt(row)
            tokens=[1,*tokenizer.encode(text).ids];lengths.append(len(tokens))
            if len(tokens)+budget>config.max_text_length:raise ValueError('Complete unchanged original held prompt plus fixed generation budget does not fit')
            records.append(dict(benchmark=benchmark,id=str(row['id']),prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                prompt_tokens=len(tokens),native_input_ids_sha256=hashlib.sha256(np.asarray(tokens,dtype='<u2').tobytes()).hexdigest(),
                fixed_generation_budget=budget,optimization_forbidden=True))
        summary[benchmark]=dict(original_held_questions=count,minimum_prompt_tokens=min(lengths),maximum_prompt_tokens=max(lengths),
            complete_prompt_tokens=sum(lengths),fixed_generation_budget=budget,overlength_questions=0)
        if benchmark=='MMLU-Pro':summary[benchmark]['original_domain_counts']=dict(Counter(r['domain'] for r in rows))
    OUT.mkdir();records_path=OUT/'original-native-input-bindings.jsonl'
    records_path.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in records))
    files=[Path(__file__),proof_path,phase_path,token_path,pool/'manifest.json',mmlu,math,nq,records_path,
        ROOT/'src/slm_lab/knowledge_chat_diagnostic_v1.py',ROOT/'src/slm_lab/math_corpus.py',ROOT/'src/slm_lab/training.py',
        ROOT/'src/slm_lab/held_question_registry_v2.py']
    bindings={str(p):file_hash(p) for p in files};bindings[str(path)]=proof['source_files_sha256'][str(path)]
    report=dict(at=datetime.now(timezone.utc).isoformat(),complete=True,own_native_context_positions=config.max_text_length,
        benchmark_inputs=summary,all_16961_original_held_QA_inputs_fit_complete_unchanged_budgets=True,
        no_context_clipping_or_question_skipping=True,source_overlap_audit_still_required=True,
        architecture_settings_only_no_neural_network_constructed=True,source_files_sha256=bindings,
        model_generations=0,model_updates=0,optimization_forbidden=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(OUT/'complete.json',report);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
