"""Bounded CPU controls on two committed original TRAIN candidate image pairs.

The original source is unfinalized: this creates no admitted training rows,
performs no optimization and certifies no held-out or general capability.
"""
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer
import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig
from slm_lab.training import Encoder,load_checkpoint
from slm_lab.vqa_pair_admission import reconstruct_question,verified_photo
from slm_lab.vqa_pair_evaluation import text,human_agreement
from slm_lab.vqa_pairs import canonical_question

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/vqa-real-pairs-v1'
SNAPSHOT=ROOT/'reports/committed-paired-photo-audit-v1'
OUT=ROOT/'reports/paired-image-dependence-CPU-v1'


def main():
    torch.set_num_threads(8)
    if OUT.exists():raise ValueError('New distinct diagnostic directory required; no overwrite/restart')
    audited=json.loads((SNAPSHOT/'results.json').read_text())
    if audited['passed'] is not True or file_hash(SNAPSHOT/'committed-prefix.jsonl')!=audited['prefix_sha256']:
        raise ValueError('Verified committed-prefix audit required')
    journal=list(map(json.loads,(SNAPSHOT/'committed-prefix.jsonl').read_text().splitlines()))
    images={(r['source_subset'],r['coco_image_id']):r for r in journal}
    source=ROOT/'data/vqa-source-cache-v1/candidate-pairs.train.jsonl'
    if audited['candidate_files_sha256'][str(source)]!=file_hash(source):raise ValueError('Original candidate metadata changed')
    candidates=list(map(json.loads,source.read_text().splitlines()))
    chosen=[];used=set();questions=set()
    for pair in sorted(candidates,key=lambda r:hashlib.sha256(r['id'].encode()).digest()):
        sides=[pair[s] for s in ['first','second']];keys=[(pair['source_subset'],s['image']['id']) for s in sides]
        if any(k not in images for k in keys):continue
        photos={images[k]['group_id'] for k in keys};question=canonical_question(pair)
        if (len(photos)!=2 or photos&used or question in questions or {s['answer'] for s in sides}!={'yes','no'}
                or any(s['exact_human_votes']<9 or s['annotation']['answer_type']!='yes/no' for s in sides)):continue
        chosen.append(pair);used|=photos;questions.add(question)
        if len(chosen)==2:break
    if len(chosen)!=2:raise ValueError('Two distinct complete high-consensus original training candidate pairs required')
    token=ROOT/'data/tokenizer-corpus/tokenizer.json';tokenizer=Tokenizer.from_file(str(token))
    config=ModelConfig.load(ROOT/'configs/model.json');encoder=Encoder(tokenizer,config,DATA,448)
    datasets=[];files={str(source):file_hash(source),str(token):file_hash(token),str(SNAPSHOT/'results.json'):file_hash(SNAPSHOT/'results.json'),
        str(SNAPSHOT/'committed-prefix.jsonl'):file_hash(SNAPSHOT/'committed-prefix.jsonl'),str(Path(__file__).resolve()):file_hash(__file__)}
    for pair in chosen:
        rows=[]
        for side in ['first','second']:
            record=images[(pair['source_subset'],pair[side]['image']['id'])]
            verified_photo(DATA,record);rows.append(reconstruct_question(pair,side,record,encoder))
            files[str(DATA/record['image'])]=record['stored_image_sha256'];files[str(DATA/record['attribution_page'])]=record['attribution_page_sha256']
        if encoder.prompt(rows[0])!=encoder.prompt(rows[1]):raise ValueError('Complete actual model questions must match')
        datasets.append(dict(pair=pair,rows=rows))
    for name in ['model','training','vqa_pair_admission','vqa_pair_evaluation','vqa_attribution','diverse_images','ocr_evaluation']:
        p=ROOT/'src/slm_lab'/(name+'.py');files[str(p)]=file_hash(p)
    checkpoints={'joint':ROOT/'checkpoints/extended-curriculum-v1/joint/model.pt',
        'broad_code936':ROOT/'checkpoints/dialogue-retention-v1/broad-code/model.pt'}
    expected={'joint':'2dacd490e4a6ed20527a8393629d560e7230c1b6fdec54c1a49dc2eedf7f4e49',
        'broad_code936':'743363d2792c2726352addfe9057649ffa3afe557707e8d82fed0f1bc69f9347'}
    for name,p in checkpoints.items():
        if file_hash(p)!=expected[name]:raise ValueError('Exact retained own completed checkpoint required')
        files[str(p)]=expected[name]
    guard=FileBindings(files);OUT.mkdir();atomic_json(OUT/'selection.json',dict(
        selected_original_candidate_ids=[x['pair']['id'] for x in datasets],source_subset='train2014',selection_uses_no_model_scores=True,
        before_final_alias_perceptual_and_validation_screens=True,training_admitted=False,files_sha256=files))
    reports={}
    with torch.no_grad():
        for name,path in checkpoints.items():
            guard.check();model,metadata=load_checkpoint(path,'cpu',training_state=False);model.eval()
            if model.config!=config or metadata['tokenizer_sha256']!=file_hash(token):raise ValueError('Exact own config/tokenizer required')
            results=[]
            for entry in datasets:
                rows=entry['rows'];pair_id=entry['pair']['id'];pixels=[encoder.pixels(row).unsqueeze(0) for row in rows]
                for i,row in enumerate(rows):
                    prompt=encoder.prompt(row);ids=torch.tensor([[encoder.bos]+encoder.tokens(prompt)])
                    batch=encoder.batch([row],'response','cpu');answers=row['original_human_annotation']['answers']
                    result=dict(pair_id=pair_id,id=row['id'],expected=row['response'],exact_human_votes=row['exact_human_agreement'],
                        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),context_tokens=batch['input_ids'].shape[1],
                        target_tokens=int((batch['labels'][:,1:]!=-100).sum()))
                    for condition,image in [('proper',pixels[i]),('counterpart',pixels[1-i]),('blank',torch.zeros_like(pixels[i]))]:
                        value=float(model(**{**batch,'pixel_values':image})['loss'])
                        generated=model.generate(ids,image,max_new_tokens=8,eos_token_id=encoder.eos)
                        answer=tokenizer.decode(generated[0,ids.shape[1]:].tolist())
                        result[condition]=dict(mean_answer_EOS_nll=value,generated=answer,
                            exact=text(answer)==text(row['response']),human_agreement=human_agreement(answer,answers))
                    result['image_conditioned_NLL_gap']=result['counterpart']['mean_answer_EOS_nll']-result['proper']['mean_answer_EOS_nll']
                    results.append(result)
                if results[-1]['blank']['generated']!=results[-2]['blank']['generated']:
                    raise ValueError('Identical complete question/blank pixels must produce identical greedy answers')
                print(name,'paired controls complete',pair_id,flush=True)
            if encoder.truncated:raise ValueError('Whole original paired questions/answers must remain complete')
            summary=dict(questions=4,pairs=2,proper_exact=sum(r['proper']['exact'] for r in results),
                counterpart_matches_original_target=sum(r['counterpart']['exact'] for r in results),
                blank_matches_original_target=sum(r['blank']['exact'] for r in results),
                fully_correct_proper_pairs=sum(all(r['proper']['exact'] for r in results if r['pair_id']==p['pair']['id']) for p in datasets),
                mean_image_conditioned_NLL_gap=sum(r['image_conditioned_NLL_gap'] for r in results)/4,
                mean_proper_answer_EOS_nll=sum(r['proper']['mean_answer_EOS_nll'] for r in results)/4)
            report=dict(checkpoint_sha256=expected[name],source_selection_sha256=file_hash(OUT/'selection.json'),
                inference_device='cpu',precision='fp32',image_size=448,max_new_tokens=8,summary=summary,results=results,
                model_updates=0,training_admitted=False,capability_gain_claim=False,goal_achieved=False)
            atomic_json(OUT/(name+'.json'),report);reports[name]=summary
            del model,metadata;gc.collect()
    guard.check(full=True)
    result=dict(observed_at=datetime.now(timezone.utc).isoformat(),selection_sha256=file_hash(OUT/'selection.json'),summaries=reports,
        actual_complete_own706M_CPU_forward_and_greedy_controls=True,identical_images_questions_precision_and_budget=True,
        original_source_files_sha256=files,model_updates=0,training_admitted=False,goal_achieved=False,
        scope='Two reused high-consensus binary original TRAIN candidate pairs from a committed photo prefix. Partial source, tiny descriptive diagnostic; not final admission, independent held-out performance, causal training comparison or high-end capability.')
    atomic_json(OUT/'comparison.json',result);print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
