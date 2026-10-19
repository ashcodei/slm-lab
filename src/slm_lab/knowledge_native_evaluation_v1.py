"""Controlled native-format diagnostic on the same complete broad-domain QA sample."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import math
from pathlib import Path
import re
import torch
from tokenizers import Tokenizer
from .corpus import file_hash
from .training import load_checkpoint
from .evaluation_provenance import ROOT, capture, seal


def interval(correct,total):
    if not total:return None
    z=1.96;p=correct/total;d=1+z*z/total
    center=(p+z*z/(2*total))/d
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [max(0,center-radius),min(1,center+radius)]


def letter(text,count):
    found=re.match(r'\s*(?:(?:the\s+)?answer(?:\s+is|:)\s*)?([A-J])\b',text,re.I)
    index=ord(found[1].upper())-65 if found else None
    return index if index is not None and index<count else None

def choice_scores(model,tokenizer,initial,count,device,prefix):
    scores=[]
    for i in range(count):
        candidate=tokenizer.encode(prefix+chr(65+i)).ids;state=initial;score=0.
        if not candidate:raise ValueError('Every candidate letter must have actual tokens')
        for position,token in enumerate(candidate):
            score+=float(state['logits'][0,-1].float().log_softmax(-1)[token])
            if position+1<len(candidate):state=model.decode(torch.tensor([[token]],device=device),state['cache'])
        scores.append(score)
    return scores


@torch.no_grad()
def evaluate(checkpoint,data,output,device='cuda',max_examples=1000,max_new_tokens=32):
    files=[ROOT/name for name in capture()['files_sha256']]+[Path(__file__)]
    provenance=capture(files);data_digest=file_hash(data)
    torch.set_num_threads(2);model,meta=load_checkpoint(checkpoint,device,training_state=False);model.eval()
    tp=Path(checkpoint).parent/'tokenizer.json';tok=Tokenizer.from_file(str(tp))
    if file_hash(tp)!=meta['tokenizer_sha256']:raise ValueError('Tokenizer mismatch')
    path=Path(data);audit=path.with_suffix('.audit.json')
    if not audit.exists() or json.loads(audit.read_text())['clean_questions_sha256']!=file_hash(path):
        raise ValueError('Use the benchmark questions with a completed training-contamination audit.')
    rows=[json.loads(line) for line in path.read_text().splitlines()];results=[];groups=defaultdict(list);skipped=[]
    amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    for row in rows[:max_examples]:
        options='\n'.join(chr(65+i)+'. '+text for i,text in enumerate(row['options']))
        question_text=f'The following is a multiple-choice question about {row["domain"]}.\n\n{row["question"]}\n\n{options}\n\nAnswer with one letter only.\nAnswer:'
        prompt='User: '+question_text+'\nAssistant:'
        tokens=[tok.token_to_id('<bos>')]+tok.encode(prompt).ids
        if len(tokens)+max_new_tokens>model.config.max_text_length:
            skipped.append({'id':row['id'],'reason':'Question exceeds context budget'});continue
        ids=torch.tensor([tokens],device=device);scores=[]
        with amp():
            initial=model.prefill(ids)
            scores=choice_scores(model,tok,initial,len(row['options']),device,' ')
            native_scores=choice_scores(model,tok,initial,len(row['options']),device,'')
            generated=model.generate(ids,max_new_tokens=max_new_tokens,eos_token_id=tok.token_to_id('<eos>'),use_cache=True)
        answer=tok.decode(generated[0,len(tokens):].tolist());prediction=letter(answer,len(row['options']))
        likelihood=max(range(len(scores)),key=lambda i:scores[i]);gold=int(row['answer_index'])
        native_likelihood=max(range(len(native_scores)),key=lambda i:native_scores[i])
        result={'id':row['id'],'domain':row['domain'],'answer_index':gold,'generated_response':answer,
            'generated_index':prediction,'generated_correct':prediction==gold,
            'likelihood_index':likelihood,'likelihood_correct':likelihood==gold,'candidate_scores':scores,
            'native_letter_likelihood_index':native_likelihood,'native_letter_likelihood_correct':native_likelihood==gold,
            'native_letter_candidate_scores':native_scores,
            'chance_accuracy':1/len(scores),'context_tokens':len(tokens)}
        results.append(result);groups[row['domain']].append(result)
        if len(results)%20==0:print('evaluated knowledge questions',len(results),flush=True)
    def summarize(items):
        total=len(items);gen=sum(r['generated_correct'] for r in items);choice=sum(r['likelihood_correct'] for r in items)
        return {'questions':total,'generated_accuracy':gen/total if total else None,
            'likelihood_accuracy':choice/total if total else None,
            'native_letter_likelihood_accuracy':sum(r['native_letter_likelihood_correct'] for r in items)/total if total else None,
            'native_letter_likelihood_accuracy_95ci':interval(sum(r['native_letter_likelihood_correct'] for r in items),total),
            'generated_correct_when_formatted':gen/sum(r['generated_index'] is not None for r in items) if any(r['generated_index'] is not None for r in items) else None,'format_compliance':sum(r['generated_index'] is not None for r in items)/total if total else None,
            'chance_accuracy':sum(r['chance_accuracy'] for r in items)/total if total else None,
            'generated_accuracy_95ci':interval(gen,total),'likelihood_accuracy_95ci':interval(choice,total)}
    report={'checkpoint_sha256':file_hash(checkpoint),'questions_sha256':file_hash(data),'generation_budget':max_new_tokens,
        'contamination_audit_sha256':file_hash(audit),'overall':summarize(results),
        'by_domain':{k:summarize(v) for k,v in groups.items()},'predictions':results,'skipped':skipped,
        'protocol':'Predeclared native User/Assistant wrapping of the unchanged full zero-shot question body; greedy cached BF16,32-token limit. Separate spaced-letter and direct-letter likelihood scores are both reported. Controlled prompt/tokenization diagnostic, preserving the raw-completion baseline.',
        'limitations':['A balanced MMLU-Pro sample; not the complete standardized leaderboard protocol.',
            'Exact question-text matches were excluded; near-duplicates/semantic overlap remain possible.',
            'Multiple-choice accuracy is not proof of reliable real-world medical, legal or other advice.']}
    seal(report,provenance,files=files,data=data,digest=data_digest,device=device,tokenizer_sha256=meta['tokenizer_sha256'])
    Path(output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='predictions'},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--max-examples',type=int,default=1000);a=p.parse_args()
    evaluate(a.checkpoint,a.data,a.output,a.device,a.max_examples)

if __name__=='__main__':main()
