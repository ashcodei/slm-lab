"""Keep complete original human preference pairs and reserve knowledge questions."""
from datetime import datetime,timezone
import json
from pathlib import Path
import ahocorasick
from tokenizers import Tokenizer
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json,jsonl
from slm_lab.model import ModelConfig
from slm_lab.training import Encoder

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'data/human-preference-curriculum-v1';output.mkdir(exist_ok=True)
    tp=ROOT/'data/tokenizer-corpus/tokenizer.json';questions=ROOT/'data/knowledge-eval-v1/questions.jsonl'
    encoder=Encoder(Tokenizer.from_file(str(tp)),ModelConfig.load(ROOT/'configs/model.json'),output)
    matcher=ahocorasick.Automaton()
    for row in jsonl(questions):matcher.add_word(' '.join(row['question'].casefold().split()),row['id'])
    matcher.make_automaton();sources=[];excluded=[];counts={};kept={}
    heldout={row['id'] for row in jsonl(ROOT/'data/alignment-public-v1/preferences.valid.jsonl')}
    for split in ['train','valid']:
        source=ROOT/'data/alignment-public-v1'/('preferences.'+split+'.jsonl')
        sources.append({'path':str(source),'sha256':file_hash(source)})
        records=[]
        for row in jsonl(source):
            if row.get('feedback_source')!='human':raise ValueError('Expected exclusively human-ranked source data')
            reason=None
            if split=='train' and row['id'] in heldout:reason='held-out source prompt'
            text=' '.join((row['prompt']+' '+row['chosen']+' '+row['rejected']).casefold().split())
            matches=[key for _,key in matcher.iter(text)]
            if matches:reason='reserved knowledge question'
            previous=encoder.truncated;encoder.sequence(row,'chosen');encoder.sequence(row,'rejected')
            if encoder.truncated!=previous:reason='pair would lose original prompt or completion text'
            if reason:
                excluded.append({'id':row['id'],'split':split,'reason':reason,'benchmark_question_ids':matches});continue
            records.append(row)
        kept[split]=records;counts['preferences.'+split]=len(records)
        target=output/('preferences.'+split+'.jsonl')
        target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        # Reserve generation space without silently deleting any original prompt tokens.
        prompts=[{'id':r['id'],'prompt':r['prompt'],'source':r['source'],'source_preference_id':r['id'],
                  'prompt_origin':'Source prompt from a human-ranked pair; rollout responses come from our own policy.'}
                 for r in records if len(encoder.tokens(encoder.prompt(r)))+257<=encoder.config.max_text_length]
        (output/('ppo.'+split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in prompts))
        counts['ppo.'+split]=len(prompts)
    if {r['id'] for r in kept['train']} & {r['id'] for r in kept['valid']}:raise ValueError('Preference prompt overlap')
    report={'created_at':datetime.now(timezone.utc).isoformat(),'splits':counts,'source_files':sources,
            'source_manifest_sha256':file_hash(ROOT/'data/alignment-public-v1/manifest.json'),
            'tokenizer_sha256':file_hash(tp),'benchmark_questions_sha256':file_hash(questions),
            'files_sha256':{p.name:file_hash(p) for p in output.glob('*.jsonl')},'excluded':excluded,
            'heldout_source_prompt_ids_protected':len(heldout),'max_rollout_tokens_reserved':256,
            'limitations':['Human ranking labels are original HH labels for other-model responses, not fresh labels for our generated responses.',
                           'Exact benchmark questions and selected source prompt IDs checked; semantic overlap remains possible.',
                           'Rejected/chosen responses retain their original model-generated origin and human-ranked status.']}
    atomic_json(output/'manifest.json',report);print(json.dumps({'splits':counts,'excluded':len(excluded)},indent=2))

if __name__=='__main__':main()
